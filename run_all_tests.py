"""
Single entry point for the whole test suite. Runs every test_*.py under
tests/ as a subprocess and reports pass/fail per file plus a total.

Test files moved from repo root into tests/ on 2026-07-14; this file stays
at repo root as the entry point so `python run_all_tests.py` keeps working
unchanged. Each subprocess runs with this file's own (repo root) working
directory, which the moved tests' `sys.path.insert(0, "src")` calls and any
relative data/ paths still depend on - always run this from repo root.

Run: python run_all_tests.py
Exit 0 = every file passed, exit 1 = at least one failed.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEST_FILES = sorted((HERE / "tests").glob("test_*.py"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = {}
for f in TEST_FILES:
    proc = subprocess.run([sys.executable, str(f)], capture_output=True, text=True)
    results[f.name] = (proc.returncode == 0, proc.stdout, proc.stderr)

print("=" * 60)
print("TEST SUITE SUMMARY")
print("=" * 60)
any_failed = False
for name, (ok, out, err) in results.items():
    tag = PASS if ok else FAIL
    # Pull the "N/M passed" line out of stdout if present, for a quick count.
    summary_line = ""
    for line in out.splitlines()[::-1]:
        if "passed" in line:
            summary_line = f" ({line.strip()})"
            break
    print(f"  [{tag}] {name}{summary_line}")
    if not ok:
        any_failed = True
        print(f"    --- stdout tail ---")
        for line in out.splitlines()[-15:]:
            print(f"    {line}")
        if err.strip():
            print(f"    --- stderr tail ---")
            for line in err.splitlines()[-15:]:
                print(f"    {line}")

print("=" * 60)
if any_failed:
    print("FAILURES FOUND - see details above.")
else:
    print("All test files passed.")

sys.exit(1 if any_failed else 0)
