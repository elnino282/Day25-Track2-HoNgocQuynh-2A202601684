"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 additional_markdown: str | None = None) -> str:
    """Return a markdown cost-optimization report."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cleanest region: {sustainability.get('cleanest_region', sustainability.get('best_region', 'n/a'))}",
            f"- Cheapest-electricity region: {sustainability.get('cheapest_region', 'n/a')}",
            f"- Balanced cost/carbon region: {sustainability.get('balanced_region', 'n/a')}",
        ]
    if additional_markdown:
        lines += ["", additional_markdown.strip()]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(
    levers: dict,
    path: str,
    baseline_usd: float | None = None,
    optimized_usd: float | None = None,
) -> str:
    """Write a true baseline-to-optimized waterfall chart PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    baseline = float(baseline_usd if baseline_usd is not None else sum(levers.values()))
    optimized = float(
        optimized_usd if optimized_usd is not None else baseline - sum(levers.values())
    )
    names = ["Baseline", *list(levers.keys()), "Optimized"]
    heights = [baseline]
    bottoms = [0.0]
    running = baseline
    for amount in levers.values():
        running -= amount
        heights.append(amount)
        bottoms.append(running)
    heights.append(optimized)
    bottoms.append(0.0)
    colors = ["#315a8a", *(["#2a9d8f"] * len(levers)), "#5b3f8c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, heights, bottom=bottoms, color=colors, width=0.72)
    for i, bar in enumerate(bars):
        value = baseline if i == 0 else optimized if i == len(bars) - 1 else heights[i]
        label = f"${value:,.0f}" if i in (0, len(bars) - 1) else f"-${value:,.0f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bottoms[i] + heights[i] + baseline * 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Monthly spend (USD)")
    ax.set_title("NimbusAI GPU cost optimization waterfall")
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
