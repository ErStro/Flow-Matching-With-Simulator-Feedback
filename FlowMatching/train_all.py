import os

from baseline_net.train_baseline_net import BaselineNetTrainer
from WhiteBoxSimulatorFeedback.train_refinement_net import RefinementTrainer
from blackbox.train_blackbox_net import BlackBoxTrainer


def main():
    base_dir = os.path.dirname(__file__)
    baseline_dir = os.path.join(base_dir, "baseline_net")
    whitebox_dir = os.path.join(base_dir, "WhiteBoxSimulatorFeedback")
    blackbox_dir = os.path.join(base_dir, "blackbox")

    baseline_1000_path = os.path.join(baseline_dir, "baseline_model_1000.pt")
    baseline_1500_path = os.path.join(baseline_dir, "baseline_model_1500.pt")

    baseline_trainer_1000 = BaselineNetTrainer(epochs=1000)
    baseline_trainer_1000.train(save_path=baseline_1000_path)

    baseline_trainer_1500 = BaselineNetTrainer(epochs=1500)
    baseline_trainer_1500.train(save_path=baseline_1500_path)

    whitebox_trainer = RefinementTrainer(
        baseline_model_path=baseline_1000_path,
        baseline_model_file=os.path.join(baseline_dir, "flow_matching_model.py"),
        epochs=500,
        save_path=os.path.join(whitebox_dir, "refinement_model_500.pt"),
    )
    whitebox_trainer.train()

    blackbox_trainer = BlackBoxTrainer(
        baseline_model_path=baseline_1000_path,
        baseline_model_file=os.path.join(baseline_dir, "flow_matching_model.py"),
        epochs=500,
        save_path=os.path.join(blackbox_dir, "blackbox_model_500.pt"),
    )
    blackbox_trainer.train()


if __name__ == "__main__":
    main()
