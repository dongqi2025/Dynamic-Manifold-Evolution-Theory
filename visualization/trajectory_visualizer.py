import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_trajectory(processed, config, seq_id, highlight_steps=None):
    traj = processed.cpu().numpy()

    if traj.ndim == 3:
        traj = traj.squeeze(0)

    plt.figure(figsize=(12, 6))

    if config.pca_components >= 3:
        ax = plt.subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.5)
        sc = ax.scatter(traj[::5, 0], traj[::5, 1], traj[::5, 2],
                        c=range(len(traj[::5])), cmap='viridis', s=20)
        ax.set_zlabel("PC3")
    else:
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)
        sc = plt.scatter(traj[::5, 0], traj[::5, 1],
                         c=range(len(traj[::5])), cmap='viridis', s=20)

    if highlight_steps:
        for step in highlight_steps:
            if step < len(traj):
                if config.pca_components >= 3:
                    ax.scatter(traj[step, 0], traj[step, 1], traj[step, 2],
                               c='red', s=100, marker='x')
                else:
                    plt.scatter(traj[step, 0], traj[step, 1],
                                c='red', s=100, marker='x')

    plt.title("State Space Trajectory")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(sc, label="Time Step")

    plt.savefig(os.path.join(config.output_dir, f"state_trajectory_seq{seq_id}.png"))
    plt.close()
