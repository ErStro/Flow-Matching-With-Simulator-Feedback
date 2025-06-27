import matplotlib.pyplot as plt


def plot_training_metrics(weight_drift_log, cosine_sim_log):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(weight_drift_log, label="Weight Drift (L2)", color="tab:blue")
    ax2.plot(cosine_sim_log, label="Cosine Similarity", color="tab:orange")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("L2 Drift", color="tab:blue")
    ax2.set_ylabel("Cosine Sim", color="tab:orange")
    ax1.set_title("Parameteränderung über das Training")
    ax1.grid(True)

    fig.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
