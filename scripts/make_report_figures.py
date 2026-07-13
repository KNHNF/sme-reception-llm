"""
Build the report and viva figures from the evaluation artefacts.

Outputs (evaluation/figures/):
  fig_finetune_gain.png    vanilla vs fine-tuned action accuracy (GPU eval, 480-record synthetic set)
  fig_size_sweep.png       accuracy and latency across the six-model size sweep (two panels, aligned harness)
  fig_real_audio.png       strict vs scope-aware accuracy on real recordings, per model
  fig_project_timeline.png project workstreams, owners, and numbered milestones
  fig_whisper_tradeoff.png accuracy and total latency (STT+LLM), small vs tiny, all six models

Data sources: evaluation/cpu_results/aligned_summary.json and
evaluation/real_audio_results/real_audio_summary.json. The fine-tune gain
numbers are the GPU evaluation results (evaluate_model.py); that run used the
480-record synthetic training set (verified 2026-07-10: its expected outputs
exactly match sme_train.jsonl), so they are training-set accuracy and are
labelled as such; the held-out check is the aligned harness. Timeline dates
come from project records (Kanban screenshot 18 May, Teams decision 7 Jun,
eval artefact timestamps, module deadlines).

Colour use: one accent blue for the entity under discussion, grey strictly for
de-emphasised context (always with direct labels), and a validated blue/orange
pair for the one genuinely two-series chart. No dual axes, bars always start
at zero; the bounded accuracy panel uses dots, not truncated bars.

Usage: python scripts/make_report_figures.py
"""

import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
FIG = ROOT / "evaluation" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE = "#1f77b4"
ORANGE = "#d97706"
GREY = "#9aa0a6"
LIGHT = "#c7d7e6"
INK = "#333333"
MUTED = "#666666"

MODEL_LABEL = {
    "phi3": "Phi-3 mini\n3.8B",
    "llama3": "Llama 3.2\n3B",
    "qwen1.5b": "Qwen 2.5\n1.5B",
    "llama1b": "Llama 3.2\n1B",
    "qwen0.5b": "Qwen 2.5\n0.5B",
    "smol360": "SmolLM2\n360M",
}
SWEEP_ORDER = ["phi3", "llama3", "qwen1.5b", "llama1b", "qwen0.5b", "smol360"]
DEPLOYED = "qwen0.5b"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def style_ax(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)


def titles(fig, title, subtitle):
    fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.915, subtitle, fontsize=9, color=MUTED, ha="left")


def fig_finetune_gain():
    labels = ["Phi-3 mini\nvanilla", "Phi-3 mini\nfine-tuned", "Llama 3.2 3B\nvanilla", "Llama 3.2 3B\nfine-tuned"]
    vals = [0.4, 98.1, 0.0, 99.8]
    colors = [GREY, BLUE, GREY, BLUE]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor="white", linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Action accuracy (%)", color=INK)
    style_ax(ax)
    titles(fig, "Fine-tuning is what makes the task work",
           "GPU evaluation, 480-record synthetic set (the training data); held-out 60-record test confirms 98.3-100%")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "fig_finetune_gain.png", dpi=200)
    plt.close(fig)


def fig_size_sweep():
    rows = json.loads((ROOT / "evaluation" / "cpu_results" / "aligned_summary.json").read_text())
    best = {}
    for r in rows:
        if r.get("harness") == "aligned_sme_test" and r.get("quant") == "Q4_K_M":
            best[r["model"]] = r
    models = [m for m in SWEEP_ORDER if m in best]
    acc = [best[m]["action_acc"] for m in models]
    lat = [best[m]["latency_p50_ms"] for m in models]
    labels = [MODEL_LABEL[m] for m in models]
    x = list(range(len(models)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18})
    ax1.axhline(100, color=MUTED, linewidth=0.6, alpha=0.4)
    for i, m in enumerate(models):
        deployed = m == DEPLOYED
        ax1.plot(i, acc[i], "o", markersize=10,
                 markerfacecolor=BLUE if deployed else "white",
                 markeredgecolor=BLUE, markeredgewidth=1.6)
        ax1.annotate(f"{acc[i]:.1f}", (i, acc[i]), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9, color=INK)
    ax1.set_ylim(91, 103)
    ax1.set_ylabel("Action accuracy (%)", color=INK)
    style_ax(ax1)

    bars = ax2.bar(x, lat, width=0.55, edgecolor="white", linewidth=2,
                   color=[BLUE if m == DEPLOYED else LIGHT for m in models])
    for b, v in zip(bars, lat):
        ax2.text(b.get_x() + b.get_width() / 2, v + 120, f"{int(v)}", ha="center", fontsize=9, color=INK)
    ax2.set_ylim(0, 7000)
    ax2.set_ylabel("P50 latency (ms)", color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    style_ax(ax2)

    titles(fig, "Model size sweep, Q4_K_M on CPU (60-record aligned test set)",
           "Accuracy holds from 3.8B down to 0.5B and breaks at 360M; the deployed Qwen 2.5 0.5B is the filled marker and solid bar")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "fig_size_sweep.png", dpi=200)
    plt.close(fig)


def fig_real_audio():
    rows = json.loads((ROOT / "evaluation" / "real_audio_results" / "real_audio_summary.json").read_text())
    latest = {}
    for r in rows:
        latest[r["family"]] = r
    models = [m for m in SWEEP_ORDER if m in latest]
    strict = [latest[m]["action_accuracy_strict"] for m in models]
    scope = [latest[m]["action_accuracy_scope_aware"] for m in models]
    n_clips = latest[models[0]]["n_clips"]
    n_speakers = latest[models[0]]["n_speakers"]
    labels = [MODEL_LABEL[m] for m in models]
    x = list(range(len(models)))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar([i - w / 2 for i in x], strict, width=w, color=BLUE,
                edgecolor="white", linewidth=2, label="Strict (exact action)")
    b2 = ax.bar([i + w / 2 for i in x], scope, width=w, color=ORANGE,
                edgecolor="white", linewidth=2, label="Scope-aware")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{b.get_height():.0f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for tick, m in zip(ax.get_xticklabels(), models):
        if m == DEPLOYED:
            tick.set_fontweight("bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    style_ax(ax)
    titles(fig, f"Real-audio evaluation ({n_clips} clips, {n_speakers} speakers, Whisper small)",
           "Synthetic accuracy does not fully survive real speech; the deployed model (bold) copes best")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "fig_real_audio.png", dpi=200)
    plt.close(fig)


def fig_project_timeline():
    lanes = [
        ("Project management, Kanban (Karan)", True, date(2026, 5, 18), date(2026, 8, 4)),
        ("Literature review (Theo, all)", False, date(2026, 5, 19), date(2026, 6, 20)),
        ("Data + fine-tuning, QLoRA (Karan)", True, date(2026, 6, 7), date(2026, 7, 5)),
        ("Backend pipeline (Karan)", True, date(2026, 6, 14), date(2026, 7, 8)),
        ("CPU deployment + evaluation (Karan)", True, date(2026, 7, 2), date(2026, 7, 12)),
        ("Real-audio collection (Karan + participants)", True, date(2026, 6, 30), date(2026, 7, 14)),
        ("Web frontend (Peter, Goodnews)", False, date(2026, 7, 1), date(2026, 7, 20)),
        ("User eval + business case (Christopher)", False, date(2026, 7, 14), date(2026, 7, 25)),
        ("Portfolio wiki (all)", False, date(2026, 7, 7), date(2026, 7, 30)),
    ]
    milestones = [
        (date(2026, 5, 18), "Kanban set up"),
        (date(2026, 6, 7), "Model decision"),
        (date(2026, 6, 19), "v1 models fine-tuned"),
        (date(2026, 6, 30), "Mock viva; voice collection opens"),
        (date(2026, 7, 5), "Six-model sweep complete"),
        (date(2026, 7, 7), "Class recording session"),
        (date(2026, 7, 20), "Group drafts due"),
        (date(2026, 7, 30), "Portfolio due"),
        (date(2026, 8, 4), "Viva"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    names = []
    for i, (name, karan, s, e) in enumerate(reversed(lanes)):
        ax.barh(i, (e - s).days, left=mdates.date2num(s), height=0.52,
                color=BLUE if karan else LIGHT, edgecolor="white", linewidth=1)
        names.append(name)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(names, fontsize=9, color=INK)

    top = len(lanes) - 0.1
    for idx, (dt, _label) in enumerate(milestones, start=1):
        xpos = mdates.date2num(dt)
        ax.axvline(xpos, color=MUTED, linewidth=0.5, alpha=0.25, zorder=0)
        row = 0 if idx % 2 else 1
        y = top + 0.55 + row * 0.75
        ax.plot(xpos, y, marker="D", markersize=9, color=INK, markerfacecolor="white",
                markeredgewidth=1.2, clip_on=False)
        ax.text(xpos, y, str(idx), ha="center", va="center", fontsize=6.5, color=INK, zorder=5)
    ax.set_ylim(-0.7, top + 2.3)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(mdates.date2num(date(2026, 5, 14)), mdates.date2num(date(2026, 8, 8)))
    ax.tick_params(axis="x", labelsize=8.5, colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.25)
    ax.set_axisbelow(True)

    key = ["{}  {} ({})".format(i, lbl, dt.strftime("%d %b")) for i, (dt, lbl) in enumerate(milestones, start=1)]
    cols = [key[0:3], key[3:6], key[6:9]]
    for c, items in enumerate(cols):
        fig.text(0.07 + c * 0.32, 0.075 - 0, "\n".join(items), fontsize=8, color=MUTED, va="top")

    fig.suptitle("Project workstreams and milestones (blue = Karan-led, light = shared or teammate-led)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(FIG / "fig_project_timeline.png", dpi=200)
    plt.close(fig)


def fig_whisper_tradeoff():
    rows = json.loads((ROOT / "evaluation" / "real_audio_results" / "real_audio_summary.json").read_text())
    by_key = {(r["family"], r["whisper"]): r for r in rows if r.get("quant", "Q4_K_M") == "Q4_K_M"}
    models = [m for m in SWEEP_ORDER if (m, "small") in by_key and (m, "tiny") in by_key]
    labels = [MODEL_LABEL[m] for m in models]
    x = list(range(len(models)))
    w = 0.36

    acc_small = [by_key[(m, "small")]["action_accuracy_strict"] for m in models]
    acc_tiny = [by_key[(m, "tiny")]["action_accuracy_strict"] for m in models]
    stt_small = [by_key[(m, "small")]["stt_latency_p50_ms"] for m in models]
    llm_small = [by_key[(m, "small")]["latency_p50_ms"] for m in models]
    stt_tiny = [by_key[(m, "tiny")]["stt_latency_p50_ms"] for m in models]
    llm_tiny = [by_key[(m, "tiny")]["latency_p50_ms"] for m in models]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.2})

    b1 = ax1.bar([i - w / 2 for i in x], acc_small, width=w, color=BLUE, edgecolor="white", linewidth=2, label="Whisper small")
    b2 = ax1.bar([i + w / 2 for i in x], acc_tiny, width=w, color=ORANGE, edgecolor="white", linewidth=2, label="Whisper tiny (deployed)")
    for bars in (b1, b2):
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}", ha="center", fontsize=8.5, color=INK)
    ax1.set_ylim(0, 38)
    ax1.set_ylabel("Strict action accuracy (%)", color=INK)
    ax1.legend(frameon=False, loc="upper left", fontsize=9)
    style_ax(ax1)

    qi = models.index("qwen0.5b")
    ax1.annotate("best on accuracy\nat both sizes", xy=(qi - w / 2 - 0.05, acc_small[qi] - 4), xytext=(qi - 1.5, 33),
                fontsize=8.5, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                                connectionstyle="arc3,rad=0.2"))

    bs1 = ax2.bar([i - w / 2 for i in x], stt_small, width=w, color=LIGHT, edgecolor="white", linewidth=1.5)
    bl1 = ax2.bar([i - w / 2 for i in x], llm_small, width=w, bottom=stt_small, color=BLUE, edgecolor="white", linewidth=1.5)
    bs2 = ax2.bar([i + w / 2 for i in x], stt_tiny, width=w, color="#f3d9b8", edgecolor="white", linewidth=1.5)
    bl2 = ax2.bar([i + w / 2 for i in x], llm_tiny, width=w, bottom=stt_tiny, color=ORANGE, edgecolor="white", linewidth=1.5)
    for i in x:
        tot_s = stt_small[i] + llm_small[i]
        tot_t = stt_tiny[i] + llm_tiny[i]
        ax2.text(i - w / 2, tot_s + 100, f"{tot_s/1000:.1f}s", ha="center", fontsize=8.5, color=INK)
        ax2.text(i + w / 2, tot_t + 100, f"{tot_t/1000:.1f}s", ha="center", fontsize=8.5, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    for tick, m in zip(ax2.get_xticklabels(), models):
        if m == "qwen0.5b":
            tick.set_fontweight("bold")
    ax2.set_ylabel("Total response latency (ms)\nlight = STT, dark = LLM", color=INK)
    style_ax(ax2)

    tot_t_q = (stt_tiny[qi] + llm_tiny[qi])
    ax2.annotate("fastest total\nresponse", xy=(qi + w / 2 + 0.05, tot_t_q + 300), xytext=(qi - 1.5, 3700),
                fontsize=8.5, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                                connectionstyle="arc3,rad=-0.25"))

    fig.suptitle("Whisper small vs tiny: accuracy and total latency, by model",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig_whisper_tradeoff.png", dpi=200)
    plt.close(fig)


def fig_loss_grid():
    """2x2 grid of the four existing per-step loss-curve PNGs (they carry their
    own titles), one image for the portfolio instead of two arbitrary picks."""
    panels = [
        FIG / "loss_sme-phi3-qlora.png",
        FIG / "loss_sme-qwen0.5b-qlora-v2.png",
        FIG / "loss_sme-llama3-1b-qlora-v2.png",
        FIG / "loss_sme-qwen1.5b-qlora-v2.png",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for ax, path in zip(axes.flat, panels):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG / "fig_loss_grid.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_finetune_gain()
    fig_size_sweep()
    fig_real_audio()
    fig_project_timeline()
    fig_whisper_tradeoff()
    fig_loss_grid()
    print("Wrote 6 figures to", FIG)
