"""Render reproducible charts for the current SFT run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sft"

COLORS = {
    "blue": "#2563EB",
    "blue_light": "#93C5FD",
    "green": "#16A34A",
    "green_light": "#86EFAC",
    "orange": "#EA580C",
    "orange_light": "#FDBA74",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "gray_light": "#E2E8F0",
    "ink": "#172033",
    "grid": "#D8DEE9",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_matplotlib() -> None:
    preferred = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    installed = {item.name for item in font_manager.fontManager.ttflist}
    font = next((name for name in preferred if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["gray"],
            "ytick.color": COLORS["gray"],
            "text.color": COLORS["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(RESULTS / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_value_labels(ax, bars, *, suffix="", color=None, fontsize=9) -> None:
    for bar in bars:
        value = bar.get_width() if bar.get_width() else bar.get_height()
        if bar.get_width() >= bar.get_height():
            ax.text(
                bar.get_x() + bar.get_width() + ax.get_xlim()[1] * 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{value:g}{suffix}",
                va="center",
                ha="left",
                fontsize=fontsize,
                color=color or COLORS["ink"],
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height(),
                f"{value:g}{suffix}",
                va="bottom",
                ha="center",
                fontsize=fontsize,
                color=color or COLORS["ink"],
            )


def chart_data_pipeline(dataset: dict, training: dict) -> None:
    existing = dataset["sources"]["existing"]["rows"]
    raw = dataset["sources"]["incremental"]["raw_rows"]
    accepted = dataset["sources"]["incremental"]["strict_accepted"]
    selected = dataset["sources"]["incremental"]["selected_for_merge"]
    total = dataset["split"]["total"]
    train = dataset["split"]["train"]
    validation = dataset["split"]["validation"]
    trained = training["train_examples"]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("SFT 数据构建与训练样本漏斗", pad=18, fontsize=20)

    nodes = [
        (0.4, 5.2, 2.2, 1.25, "既有成功轨迹", existing, COLORS["blue_light"]),
        (0.4, 2.5, 2.2, 1.25, "增量 raw Teacher", raw, COLORS["orange_light"]),
        (3.2, 2.5, 2.2, 1.25, "Reward v3 严格通过", accepted, COLORS["orange_light"]),
        (6.0, 2.5, 2.2, 1.25, "长度与覆盖筛选", selected, COLORS["green_light"]),
        (8.9, 4.2, 2.2, 1.4, "合并数据", total, COLORS["green_light"]),
        (11.7, 5.3, 2.0, 1.2, "Train split", train, COLORS["blue_light"]),
        (11.7, 3.55, 2.0, 1.2, "Validation", validation, COLORS["purple"]),
        (11.7, 1.45, 2.0, 1.2, "实际进入训练", trained, COLORS["green_light"]),
    ]

    for x, y, w, h, label, value, color in nodes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=COLORS["gray"],
            facecolor=color,
            alpha=0.95,
        )
        ax.add_patch(patch)
        text_color = "white" if color == COLORS["purple"] else COLORS["ink"]
        ax.text(x + w / 2, y + h * 0.68, label, ha="center", va="center", fontsize=11, color=text_color)
        ax.text(x + w / 2, y + h * 0.30, str(value), ha="center", va="center", fontsize=21, weight="bold", color=text_color)

    arrows = [
        ((2.6, 3.12), (3.2, 3.12), ""),
        ((5.4, 3.12), (6.0, 3.12), "保留 97%"),
        ((2.6, 5.82), (8.9, 5.08), ""),
        ((8.2, 3.12), (8.9, 4.55), ""),
        ((11.1, 4.9), (11.7, 5.9), ""),
        ((11.1, 4.75), (11.7, 4.15), ""),
    ]
    for start, end, label in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.8),
        )
        if label:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(mx, my + 0.22, label, ha="center", fontsize=9, color=COLORS["gray"])

    ax.annotate(
        "",
        xy=(13.72, 2.05),
        xytext=(13.72, 5.9),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["gray"],
            lw=1.8,
            connectionstyle="angle3,angleA=0,angleB=90",
        ),
    )
    ax.text(
        14.05,
        3.2,
        "token gate\n丢弃 5 条",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["gray"],
    )

    ax.text(
        7,
        0.55,
        "task_id 隔离：Train / Validation 重叠 0；与 Final-200 重叠 0",
        ha="center",
        fontsize=12,
        color=COLORS["gray"],
    )
    save(fig, "01_sft_data_pipeline.png")


def chart_training_curve(training: dict) -> None:
    history = training["eval_history"]
    epochs = [item["epoch"] for item in history]
    losses = [item["eval_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(epochs, losses, marker="o", markersize=10, linewidth=3, color=COLORS["blue"])
    ax.fill_between(epochs, losses, min(losses) - 0.012, color=COLORS["blue_light"], alpha=0.25)
    ax.axhline(training["train_loss"], color=COLORS["green"], linestyle="--", linewidth=2, label=f"最终 train loss = {training['train_loss']:.4f}")

    for item in history:
        ax.annotate(
            f"{item['eval_loss']:.4f}\nstep {item['step']}",
            (item["epoch"], item["eval_loss"]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.set_title("Validation Loss 随训练轮次持续下降", pad=16, fontsize=19)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.set_ylim(min(training["train_loss"], min(losses)) - 0.015, max(losses) + 0.025)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.01,
        -0.18,
        f"3 epochs · 177 optimizer steps · 训练 {training['total_time_minutes']:.1f} 分钟 · 峰值显存 {training['peak_gpu_memory_gib']:.2f} GiB",
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["gray"],
    )
    save(fig, "02_sft_training_curve.png")


def chart_action_coverage(dataset: dict) -> None:
    coverage = dataset["action_coverage"]
    ordered = sorted(coverage, key=lambda name: coverage[name]["combined"])
    existing = [coverage[name]["existing"] for name in ordered]
    incremental = [coverage[name]["incremental"] for name in ordered]
    totals = [coverage[name]["combined"] for name in ordered]

    labels = [name.replace("_", "\n") for name in ordered]
    y = range(len(ordered))
    fig, ax = plt.subplots(figsize=(13, 8))
    base = ax.barh(y, existing, color=COLORS["blue_light"], label="既有轨迹")
    added = ax.barh(y, incremental, left=existing, color=COLORS["orange"], label="本次增量")

    for idx, total in enumerate(totals):
        ax.text(total + max(totals) * 0.012, idx, f"{total}", va="center", fontsize=9)
    for idx, value in enumerate(incremental):
        if value >= 20:
            ax.text(existing[idx] + value / 2, idx, f"+{value}", ha="center", va="center", fontsize=8, color="white", weight="bold")

    ax.set_yticks(list(y), labels)
    ax.set_xlabel("工具调用次数")
    ax.set_title("工具动作覆盖：增量数据补充了关键状态，但 next_page 仍稀缺", pad=16, fontsize=18)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, max(totals) * 1.13)
    save(fig, "03_sft_action_coverage.png")


def chart_final200(final200: dict) -> None:
    outcomes = [
        ("严格成功\ngold_purchase", final200["reward_type_counts"].get("gold_purchase", 0), COLORS["green"]),
        ("部分替代品", final200["reward_type_counts"].get("partial_alternative_purchase", 0), COLORS["orange"]),
        ("重复循环", final200["reward_type_counts"].get("repeat_loop", 0), COLORS["purple"]),
        ("错误购买", final200["reward_type_counts"].get("wrong_purchase", 0), COLORS["red"]),
        ("未知/异常终局", final200["reward_type_counts"].get("unknown", 0), COLORS["gray"]),
        ("达到最大步数", final200["reward_type_counts"].get("max_steps", 0), COLORS["blue"]),
        ("Reward 不可验证", final200["reward_type_counts"].get("reward_unverifiable", 0), COLORS["gray_light"]),
    ]
    outcomes.sort(key=lambda item: item[1])
    labels = [item[0] for item in outcomes]
    values = [item[1] for item in outcomes]
    colors = [item[2] for item in outcomes]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("任务数（共 200 条）")
    ax.set_title("Final-200 Reward V3 终局分布", pad=16, fontsize=19)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlim(0, max(values) * 1.18)

    for bar, value in zip(bars, values):
        ax.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value}  ({value / 200:.1%})", va="center", fontsize=10)

    ax.text(
        0.99,
        0.04,
        f"严格成功：{final200['strict_successes']}/200 = {final200['strict_success_rate']:.1%}\n平均工具步数：{final200['average_steps']:.2f}　Reward 可验证：{final200['reward_valid_tasks']}/200",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor=COLORS["grid"]),
    )
    save(fig, "04_sft_final200_outcomes.png")


def chart_overview(dataset: dict, training: dict, final200: dict) -> None:
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.25)
    fig.suptitle("Qwen3.5-2B SFT 1.0 关键过程与结果", fontsize=24, weight="bold", y=0.98)

    ax1 = fig.add_subplot(grid[0, 0])
    data_values = [
        dataset["sources"]["existing"]["rows"],
        dataset["sources"]["incremental"]["selected_for_merge"],
    ]
    bars = ax1.bar(["既有成功轨迹", "本次定向增量"], data_values, color=[COLORS["blue_light"], COLORS["orange"]], width=0.58)
    for bar, value in zip(bars, data_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 8, str(value), ha="center", fontsize=12, weight="bold")
    ax1.set_title("数据构成：428 + 97 = 525")
    ax1.set_ylabel("轨迹数")
    ax1.set_ylim(0, 500)
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(0.5, 0.88, "Train 473 → 实训 468\nValidation 52\nFinal-200 重叠 0", transform=ax1.transAxes, ha="center", va="top", fontsize=11, color=COLORS["gray"])

    ax2 = fig.add_subplot(grid[0, 1])
    history = training["eval_history"]
    ax2.plot([x["epoch"] for x in history], [x["eval_loss"] for x in history], marker="o", markersize=8, linewidth=3, color=COLORS["blue"])
    for item in history:
        ax2.text(item["epoch"], item["eval_loss"] + 0.006, f"{item['eval_loss']:.4f}", ha="center", fontsize=10)
    ax2.axhline(training["train_loss"], color=COLORS["green"], linestyle="--", linewidth=2)
    ax2.text(2.95, training["train_loss"] + 0.003, f"train {training['train_loss']:.4f}", ha="right", fontsize=9, color=COLORS["green"])
    ax2.set_title("3 Epoch Validation Loss")
    ax2.set_xticks([1, 2, 3])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_ylim(0.30, 0.40)
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(grid[1, 0])
    outcomes = [
        ("严格成功", 125, COLORS["green"]),
        ("部分替代", 32, COLORS["orange"]),
        ("重复循环", 26, COLORS["purple"]),
        ("其他", 17, COLORS["gray"]),
    ]
    values = [x[1] for x in outcomes]
    wedges, _ = ax3.pie(
        values,
        colors=[x[2] for x in outcomes],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white"),
    )
    ax3.text(0, 0.08, "62.5%", ha="center", va="center", fontsize=28, weight="bold", color=COLORS["green"])
    ax3.text(0, -0.17, "strict success", ha="center", va="center", fontsize=11, color=COLORS["gray"])
    ax3.set_title("Final-200 终局")
    ax3.legend(wedges, [f"{name} {value}" for name, value, _ in outcomes], frameon=False, loc="center left", bbox_to_anchor=(0.9, 0.5))

    ax4 = fig.add_subplot(grid[1, 1])
    ax4.axis("off")
    cards = [
        ("训练时长", f"{training['total_time_minutes']:.1f} 分钟", COLORS["blue_light"]),
        ("峰值显存", f"{training['peak_gpu_memory_gib']:.2f} / 94.97 GiB", COLORS["orange_light"]),
        ("优化器步数", str(training["optimizer_steps"]), COLORS["green_light"]),
        ("可训练参数", f"{training['parameters']['trainable_percent']:.3f}%", "#DDD6FE"),
    ]
    positions = [(0.02, 0.55), (0.52, 0.55), (0.02, 0.12), (0.52, 0.12)]
    for (label, value, color), (x, y) in zip(cards, positions):
        card = FancyBboxPatch((x, y), 0.44, 0.30, transform=ax4.transAxes, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor=color, edgecolor=COLORS["grid"])
        ax4.add_patch(card)
        ax4.text(x + 0.04, y + 0.21, label, transform=ax4.transAxes, fontsize=11, color=COLORS["gray"])
        ax4.text(x + 0.04, y + 0.09, value, transform=ax4.transAxes, fontsize=18, weight="bold")
    ax4.set_title("训练资源与规模", pad=12)
    ax4.text(0.5, -0.03, "BF16 · LoRA r=16 · max_length=30000 · effective batch=8", transform=ax4.transAxes, ha="center", fontsize=10, color=COLORS["gray"])

    save(fig, "00_sft_overview.png")


def chart_three_model_comparison(threeway: dict) -> None:
    results = threeway["results"]
    keys = ["base", "sft", "deepseek_v4_flash"]
    labels = ["Base\nQwen3.5-2B", "SFT\nQwen3.5-2B", "DeepSeek\nV4 Flash"]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["purple"]]

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.38, wspace=0.28)
    fig.subplots_adjust(top=0.88, bottom=0.18)
    fig.suptitle("Base、SFT 与 DeepSeek V4 Flash：Final-200 对比", fontsize=23, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.945,
        "同一冻结任务集 · Reward V3 · temperature=0 · max_steps=35",
        ha="center",
        fontsize=11,
        color=COLORS["gray"],
    )

    ax1 = fig.add_subplot(grid[0, 0])
    strict = [results[key]["strict_success_rate"] * 100 for key in keys]
    bars = ax1.bar(labels, strict, color=colors, width=0.62)
    for bar, value, key in zip(bars, strict, keys):
        count = results[key]["strict_successes"]
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 2.2, f"{count}/200\n{value:.1f}%", ha="center", fontsize=11, weight="bold")
    ax1.set_title("严格成功率（gold_purchase）")
    ax1.set_ylabel("百分比")
    ax1.set_ylim(0, 82)
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(grid[0, 1])
    rewards = [results[key]["mean_final_reward"] for key in keys]
    bars = ax2.bar(labels, rewards, color=colors, width=0.62)
    ax2.axhline(0, color=COLORS["ink"], linewidth=1)
    for bar, value in zip(bars, rewards):
        offset = 0.025 if value >= 0 else -0.05
        va = "bottom" if value >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center", va=va, fontsize=11, weight="bold")
    ax2.set_title("Mean Final Reward")
    ax2.set_ylabel("Reward")
    ax2.set_ylim(-0.2, 0.72)
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(grid[1, 0])
    x = list(range(len(keys)))
    width = 0.34
    done_rates = [results[key]["done_rate"] * 100 for key in keys]
    valid_rates = [results[key]["reward_valid_rate"] * 100 for key in keys]
    done_bars = ax3.bar([i - width / 2 for i in x], done_rates, width, color=COLORS["green"], label="环境正常 done")
    valid_bars = ax3.bar([i + width / 2 for i in x], valid_rates, width, color=COLORS["orange"], label="Reward 可验证")
    for bars_group in (done_bars, valid_bars):
        for bar in bars_group:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    ax3.set_xticks(x, labels)
    ax3.set_ylabel("百分比")
    ax3.set_ylim(0, 108)
    ax3.set_title("环境终止与 Reward 有效性")
    ax3.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend(frameon=False, loc="upper left")

    ax4 = fig.add_subplot(grid[1, 1])
    outcome_keys = [
        ("gold_purchase", "严格成功", COLORS["green"]),
        ("partial_alternative_purchase", "部分替代", COLORS["orange"]),
        ("repeat_loop", "重复循环", COLORS["purple"]),
        ("wrong_purchase", "错误购买", COLORS["red"]),
        ("max_steps", "最大步数", COLORS["blue"]),
        ("reward_unverifiable", "不可验证", COLORS["gray_light"]),
        ("unknown", "未知/异常", COLORS["gray"]),
    ]
    left = [0.0, 0.0, 0.0]
    for outcome_key, outcome_label, color in outcome_keys:
        values = [results[key]["reward_type_counts"].get(outcome_key, 0) / 2 for key in keys]
        ax4.barh(x, values, left=left, color=color, label=outcome_label)
        for idx, value in enumerate(values):
            if value >= 8:
                ax4.text(left[idx] + value / 2, idx, f"{value:.1f}%", ha="center", va="center", fontsize=8, color="white", weight="bold")
        left = [a + b for a, b in zip(left, values)]
    ax4.set_yticks(x, labels)
    ax4.set_xlim(0, 100)
    ax4.set_xlabel("任务占比")
    ax4.set_title("Reward V3 终局构成（100% 堆叠）")
    ax4.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax4.spines[["top", "right", "left"]].set_visible(False)
    legend_handles, legend_labels = ax4.get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        fontsize=8,
    )

    details = []
    for label, key in zip(labels, keys):
        flat_label = label.replace("\n", " ")
        guard = results[key]["context_projection"]["guard_rejections"]
        steps = results[key]["average_steps"]
        details.append(f"{flat_label}：平均 {steps:.2f} 步，Guard 拒绝 {guard} 次")
    fig.text(0.5, 0.025, "　|　".join(details), ha="center", fontsize=9, color=COLORS["gray"])
    save(fig, "05_final200_three_model_comparison.png")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    dataset = load_json(RESULTS / "dataset_build_metadata.json")
    training = load_json(RESULTS / "training_summary.json")
    final200 = load_json(RESULTS / "final200_summary.json")
    threeway = load_json(ROOT / "outputs" / "evaluation" / "final200-threeway-20260804.json")

    chart_overview(dataset, training, final200)
    chart_data_pipeline(dataset, training)
    chart_training_curve(training)
    chart_action_coverage(dataset)
    chart_final200(final200)
    chart_three_model_comparison(threeway)

    print("Rendered 6 SFT charts to", RESULTS)


if __name__ == "__main__":
    main()
