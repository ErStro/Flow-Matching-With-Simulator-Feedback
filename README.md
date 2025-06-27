# SBIS Flow Matching with Simulator Feedback

Re-Implements the Paper "Flow Matching for Neural Posterior Inference With Simulator Feedback" by Holzschuh et al. (https://arxiv.org/abs/2410.22573)

## TLDR
1. Run `python FlowMatching/show_it_works.py` to train baseline, whitebox and blackbox models for a few epochs (10 or 15 for the baseline, 5 for the refinement stages). The script saves the resulting models in the respective subdirectories.
2. Evaluate the saved models with `python FlowMatching/evaluate_models.py`.
3. Generate illustrative plots with `python FlowMatching/generate_plots.py`.

## Code Overview

### baseline_net
- **train_baseline_net.py**
  - `BaselineNetTrainer` prepares data and trains the baseline flow matching network.
- **flow_matching_model.py**
  - `ResidualBlock` is a simple residual MLP block.
  - `BaselineNet` combines residual blocks for the flow model.
- **flow_matching_training.py**
  - `Trainer` manages optimisation, weight drift logging and validation during training.
- **flow_matching_utils.py**
  - `FlowMatchingUtils` contains helpers for optimal transport batches and the flow-matching loss.
- **flow_matching_inference.py**
  - `Batch` dataclass describes a training batch.
  - `PosteriorSampler` integrates the learned vector field to draw posterior samples.
- **evaluate_baseline_net.py**
  - `evaluate_baseline_net` creates a corner plot for a trained baseline model.
- **training_metrics.py**
  - `plot_training_metrics` visualises parameter drift during training.

### WhiteBoxSimulatorFeedback
- **lotka_volterra_loss.py**
  - `LotkaVolterraLoss` simulates the population model and provides loss and gradient.
- **baseline_interface.py**
  - `load_baseline_model` loads a saved baseline network together with normalisation statistics.
  - `infer_theta_hat` performs one flow-matching step for an observation.
- **refinement_net.py**
  - `RefinementNet` predicts a correction based on cost information.
- **train_refinement_net.py**
  - `RefinementTrainer` trains the whitebox refinement network.
- **white_box_inference_samples.py**
  - `load_baseline`, `load_refinement`, `compute_cost`, `refine_samples`, `sample_all_observations` and `plot_samples` provide utilities for sampling and visualisation.
- **evaluate_baseline_with_sim.py**
  - `load_baseline_model`, `infer_theta_hat` and `evaluate_baseline_with_sim` evaluate the baseline with the simulator.
- **Eval.py**
  - `RefinementEvaluator` compares baseline and refinement on a dataset.
- **main.py**
  - Script demonstrating simulator driven evaluation for a single observation.
- **calculate_cost_whitebox.py**, **__init__.py**
  - Empty placeholder files.

### blackbox
- **baseline_interface.py**
  - `load_baseline_model` and `infer_theta_hat` mirror the whitebox helpers.
- **encoder.py**
  - `ObservationEncoder` encodes simulated and observed trajectories.
- **finetuning_net.py**
  - `BlackBoxRefinementNet` refines `theta_hat` using a learned controller.
- **train_blackbox_net.py**
  - `BlackBoxTrainer` performs training of encoder and refinement network.
- **evaluate_blackbox.py**
  - `evaluate_blackbox` visualises baseline versus blackbox refinement.
- **__init__.py**
  - Empty module initialiser.

### Root scripts
- **show_it_works.py** trains all components with short schedules.
- **train_all.py** runs long training schedules.
- **evaluate_models.py** samples from all models and computes C2ST, MMD and KDE based NLL metrics.
- **generate_plots.py** creates corner plots comparing baseline with whitebox and blackbox refinements.

