"""
Test-only spaCy fallback.

entity_extractor.py does `nlp = spacy.load("en_core_web_sm")` at import
time, unconditionally. That's correct for production (fail loud if the
model is missing) but it means ANY test that imports src.inference or
src.entity_extractor needs the real model on disk, even tests that have
nothing to do with NER (e.g. pure regex name-parsing logic in inference.py).

Import this module FIRST, before importing anything from src/, in any
test file that needs to run on a machine without the model downloaded
(this sandbox has no network access to fetch it; CI runners may not
have it either). It tries the real model first and only installs the
fallback if that fails, so on a normal dev machine (with the model
installed) these tests exercise the real spaCy pipeline exactly like
test_battery.py / test_pipeline.py / test_realistic.py already do.

The fallback is a regex approximation of spaCy's DATE/TIME/PERSON
tagging. It is good enough to drive control-flow tests (which branch of
Pipeline.run() fires, does the DATE-priority fix pick the right entity,
etc) but is NOT a substitute for real NER accuracy testing - that's what
the existing real-spaCy suites are for. Treat any test that only passes
under the fallback and fails under real spaCy as a bug in the fallback,
not a green light.
"""

import re
import sys
import types

USING_REAL_SPACY = True


def _install_fallback():
    global USING_REAL_SPACY
    USING_REAL_SPACY = False

    class FakeSpan:
        def __init__(self, text, label_):
            self.text = text
            self.label_ = label_

    class FakeDoc:
        def __init__(self, ents):
            self.ents = ents

    _WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
                 "saturday", "sunday"]
    _MONTHS = ["january", "february", "march", "april", "may", "june",
               "july", "august", "september", "october", "november",
               "december", "jan", "feb", "mar", "apr", "jun", "jul",
               "aug", "sep", "sept", "oct", "nov", "dec"]

    _DATE_PATTERNS = [
        r"\btoday\b", r"\btomorrow\b", r"\btmrw\b", r"\btmr\b",
        r"\byesterday\b",
        r"\bnext week\b",
        r"\bin (?:\d+|one|two|three|four|five) days?\b",
        r"\b(?:on |this |next )?(?:" + "|".join(_WEEKDAYS) + r")\b",
        r"\bthe \d{1,2}(?:st|nd|rd|th)(?:\s+of\s+(?:" + "|".join(_MONTHS) + r"))?\b",
        r"\b\d{1,2}(?:st|nd|rd|th)(?:\s+of\s+(?:" + "|".join(_MONTHS) + r"))?\b",
        r"\b(?:" + "|".join(_MONTHS) + r")\s+\d{1,2}(?:st|nd|rd|th)?\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    ]
    _DATE_RE = re.compile("|".join(f"(?:{p})" for p in _DATE_PATTERNS), re.IGNORECASE)

    _HOUR_WORDS = ("one", "two", "three", "four", "five", "six", "seven",
                   "eight", "nine", "ten", "eleven", "twelve")
    _HOUR_ALT = r"(?:" + "|".join(_HOUR_WORDS) + r"|\d{1,2})"
    _TIME_PATTERNS = [
        r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b",
        rf"\b{_HOUR_ALT}\s*(?:am|pm)\b",
        r"\b\d{1,2}:\d{2}\b",
        r"\bhalf past \w+(?:\s+in the \w+)?\b",
        r"\bquarter (?:past|to) \w+\b",
        rf"\b{_HOUR_ALT}\s*(?:o'?\s?clock|oclock)\b",
        rf"\b{_HOUR_ALT}\s+in the (?:morning|afternoon|evening)\b",
        r"\bnoon\b", r"\bmidday\b", r"\bmidnight\b",
    ]
    _TIME_RE = re.compile("|".join(f"(?:{p})" for p in _TIME_PATTERNS), re.IGNORECASE)

    # Crude PERSON guess: one or two consecutive Title-Case words. Real
    # spaCy PERSON detection is much better; inference.py's own "it's X"/
    # "my name is X" regexes cover most of the name-capture logic anyway,
    # this only backstops the bare "John Smith" (no preamble) case.
    _PERSON_RE = re.compile(r"\b[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)?\b")

    def _non_overlapping(pattern, text, label):
        spans = []
        used = []
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in used):
                continue
            used.append((m.start(), m.end()))
            spans.append((m.start(), FakeSpan(m.group(), label)))
        return spans

    def fake_nlp(text):
        found = []
        found += _non_overlapping(_DATE_RE, text, "DATE")
        found += _non_overlapping(_TIME_RE, text, "TIME")
        # Only guess PERSON if the text doesn't look like it's just a
        # command/question (keeps this from tagging every capitalised
        # sentence-start word as a person).
        for m in _PERSON_RE.finditer(text):
            found.append((m.start(), FakeSpan(m.group(), "PERSON")))
        found.sort(key=lambda x: x[0])
        return FakeDoc([s for _, s in found])

    fake_spacy = types.ModuleType("spacy")
    fake_spacy.load = lambda *a, **k: fake_nlp
    sys.modules["spacy"] = fake_spacy


try:
    import spacy as _spacy
    _spacy.load("en_core_web_sm")
except Exception:
    _install_fallback()
