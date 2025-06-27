import os
import matplotlib.pyplot as plt
import corner
import torch
import numpy as np
from sbibm.tasks import get_task

from evaluate_models import (
    sample_baseline,
    sample_whitebox,
    sample_blackbox,
    _filter_finite,
)

OBS_IDX = 6
N_SAMPLES = 200
PLOT_RANGE = [(0, 2)] * 4


def main():
    base_dir = os.path.dirname(__file__)
    baseline_dir = os.path.join(base_dir, "baseline_net")
    whitebox_dir = os.path.join(base_dir, "WhiteBoxSimulatorFeedback")
    blackbox_dir = os.path.join(base_dir, "blackbox")

    baseline_model = os.path.join(baseline_dir, "baseline_model_1500.pt")
    baseline_model_file = os.path.join(baseline_dir, "flow_matching_model.py")
    whitebox_model = os.path.join(whitebox_dir, "refinement_model_500.pt")
    blackbox_model = os.path.join(blackbox_dir, "blackbox_model_500.pt")

    task = get_task("lotka_volterra")
    theta_true = task.get_true_parameters(OBS_IDX).squeeze(0).numpy()

    base_samples = sample_baseline(
        task,
        baseline_model,
        baseline_model_file,
        OBS_IDX,
        n_samples=N_SAMPLES,
    )
    base_samples = _filter_finite(base_samples)

    wb_samples = sample_whitebox(
        task,
        baseline_model,
        baseline_model_file,
        whitebox_model,
        OBS_IDX,
        n_samples=N_SAMPLES,
    )
    wb_samples = _filter_finite(wb_samples)

    fig = corner.corner(
        base_samples.numpy(),
        color="orange",
        truths=theta_true,
        truth_color="red",
        labels=["$\\theta_0$", "$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
        show_titles=True,
        title_fmt=".2f",
        range=PLOT_RANGE,
    )
    corner.corner(
        wb_samples.numpy(),
        fig=fig,
        color="green",
        show_titles=True,
        title_fmt=".2f",
        range=PLOT_RANGE,
    )
    plt.suptitle(f"Baseline 1500 (orange) vs Whitebox (green) - Obs {OBS_IDX}")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"corner_baseline_vs_whitebox_obs{OBS_IDX}.png"))
    plt.close(fig)

    bb_samples = sample_blackbox(
        task,
        baseline_model,
        baseline_model_file,
        blackbox_model,
        OBS_IDX,
        n_samples=N_SAMPLES,
    )
    bb_samples = _filter_finite(bb_samples)

    fig = corner.corner(
        base_samples.numpy(),
        color="orange",
        truths=theta_true,
        truth_color="red",
        labels=["$\\theta_0$", "$\\theta_1$", "$\\theta_2$", "$\\theta_3$"],
        show_titles=True,
        title_fmt=".2f",
        range=PLOT_RANGE,
    )
    corner.corner(
        bb_samples.numpy(),
        fig=fig,
        color="blue",
        show_titles=True,
        title_fmt=".2f",
        range=PLOT_RANGE,
    )
    plt.suptitle(f"Baseline 1500 (orange) vs Blackbox (blue) - Obs {OBS_IDX}")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"corner_baseline_vs_blackbox_obs{OBS_IDX}.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
