"""Probability diagnostics for the notebook and reports."""

from dataclasses import replace
from typing import TYPE_CHECKING

from .data import OUTCOMES
from .evaluation import Backtest, calibration

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def calibration_plot(report: Backtest) -> "Figure":
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    quoted = tuple(row for row in report.predictions if row.fixture.market_odds is not None)
    paired = replace(report, predictions=quoted) if quoted else report
    series = [("Results model", "#106957", calibration(paired))]
    if quoted:
        series.append(("Pre-match odds", "#5346a3", calibration(paired, market=True)))

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 4.3), sharex=True, sharey=True, layout="constrained")
    for axis, outcome, title in zip(axes, OUTCOMES, ("Home win", "Draw", "Away win"), strict=True):
        axis.plot([0, 1], [0, 1], linestyle="--", color="#9aa7a1", label="Perfect calibration", zorder=1)
        for label, color, bins in series:
            points = [point for point in bins if point.outcome == outcome]
            axis.scatter(
                [point.mean_probability for point in points], [point.observed_frequency for point in points],
                s=[18 + 5 * point.matches ** 0.5 for point in points],
                color=color, label=label, alpha=0.87, zorder=2,
            )
        axis.set(title=title, xlabel="Mean forecast probability",
                 xlim=(-0.025, 1.025), ylim=(-0.025, 1.025), aspect="equal")
        axis.set_xticks([0, .25, .5, .75, 1])
        axis.set_yticks([0, .25, .5, .75, 1])
        axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(color="#e3e9e5")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Observed frequency")
    figure.suptitle(f"Retrospective reliability | {len(paired.predictions):,} fixtures", fontweight="semibold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    return figure
