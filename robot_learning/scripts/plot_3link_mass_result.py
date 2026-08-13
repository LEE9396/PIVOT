#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    trace = result["trace"]
    truth = np.asarray(result["evaluator_only_ground_truth"]["part_mass_kg"])
    estimate = np.asarray(result["final"]["part_mass_kg"])
    interval = np.asarray(result["final"]["part_mass_95_interval_kg"])
    steps = np.asarray([row["step"] for row in trace])
    history = np.asarray([row["estimate"]["part_mass_kg"] for row in trace])
    uncertainty = np.asarray(
        [
            row["uncertainty_stop"]["maximum_relative_95_half_width"] * 100
            for row in trace
        ]
    )
    condition = np.asarray(
        [
            row["uncertainty_stop"]["design_condition_number"] or np.nan
            for row in trace
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ("#3678bf", "#e07a35", "#3a9d6f")
    links = ("Link 1", "Link 2", "Link 3")

    x = np.arange(3)
    axes[0, 0].bar(x - 0.18, truth, 0.36, label="GT", color="#9aa1aa")
    axes[0, 0].bar(
        x + 0.18,
        estimate,
        0.36,
        yerr=np.vstack((estimate - interval[:, 0], interval[:, 1] - estimate)),
        capsize=5,
        label="Estimate (95% CI)",
        color=colors,
    )
    axes[0, 0].set_xticks(x, links)
    axes[0, 0].set_ylabel("Mass (kg)")
    axes[0, 0].set_title("Final part-wise mass")
    axes[0, 0].legend()

    for index, (label, color) in enumerate(zip(links, colors)):
        axes[0, 1].plot(steps, history[:, index], "o-", color=color, label=label)
        axes[0, 1].axhline(truth[index], color=color, linestyle="--", alpha=0.45)
    axes[0, 1].set_xlabel("Accepted static pose")
    axes[0, 1].set_ylabel("Mass estimate (kg)")
    axes[0, 1].set_title("Estimate convergence")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, uncertainty, "o-", color="#7a4fb3")
    axes[1, 0].axhline(5.0, color="#c43b3b", linestyle="--", label="5% target")
    axes[1, 0].set_xlabel("Accepted static pose")
    axes[1, 0].set_ylabel("Max 95% half-width (%)")
    axes[1, 0].set_title("Uncertainty")
    axes[1, 0].legend()

    axes[1, 1].plot(steps, condition, "o-", color="#2b7a78")
    axes[1, 1].axhline(50.0, color="#c43b3b", linestyle="--", label="limit 50")
    axes[1, 1].set_xlabel("Accepted static pose")
    axes[1, 1].set_ylabel("Condition number")
    axes[1, 1].set_title("Identifiability")
    axes[1, 1].legend()

    error = np.abs(estimate - truth) / truth * 100
    fig.suptitle(
        f"3-link static gravity identification | max error {error.max():.2f}% | "
        f"validation: {result['validation']['passed']}",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
