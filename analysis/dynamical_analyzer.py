import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ripser import ripser
from persim import plot_diagrams
import copy

from data_processing.data_preprocessor import DataPreprocessor


class TemporalPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, x):
        features, _ = self.lstm(x)
        return self.regressor(features)


class DynamicalAnalyzer:
    def __init__(self, config, hidden_states, seq_id=0):
        self.config = config
        self.seq_id = seq_id
        self.results = None
        preprocessor = DataPreprocessor(config)
        self.processed = preprocessor.preprocess(hidden_states)

    def analyze_continuity(self):
        processed_tensor = self.processed.float().to("cuda")
        diffs = torch.diff(processed_tensor, dim=1)
        distances = torch.norm(diffs, p=2, dim=-1).mean(dim=0)

        distances_np = distances.cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(distances_np, marker='o', linestyle='--', alpha=0.7)
        plt.title("Temporal Evolution of State Distances")
        plt.xlabel("Time Step")
        plt.ylabel("Distance")
        plt.savefig(os.path.join(self.config.output_dir, "continuity.png"))
        plt.close()

        return {
            "mean_distance": float(torch.mean(distances).item()),
            "std_distance": float(torch.std(distances).item()),
            "max_jump": float(torch.max(distances).item())
        }

    def cluster_analysis(self):
        flat_states = self.processed.reshape(-1, self.config.pca_components).cpu().numpy()

        kmeans = KMeans(n_clusters=self.config.cluster_n, random_state=42)
        labels = kmeans.fit_predict(flat_states)
        score = silhouette_score(flat_states, labels)

        plt.figure(figsize=(10, 6))
        plt.scatter(flat_states[:, 0], flat_states[:, 1],
                    c=labels, cmap='tab10', s=5, alpha=0.6)
        plt.title(f"Cluster Structure (Silhouette: {score:.2f})")
        plt.savefig(os.path.join(self.config.output_dir, "clusters.png"))
        plt.close()

        return {"silhouette_score": float(score)}

    def temporal_dynamics(self):
        X = torch.tensor(self.processed[:, :-1], dtype=torch.float32).to("cuda")
        y = torch.tensor(self.processed[:, 1:], dtype=torch.float32).to("cuda")

        model = TemporalPredictor(self.config.pca_components,
                                  self.config.lstm_hidden_dim).to("cuda")
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for epoch in range(self.config.lstm_epochs):
            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch} loss: {loss.item():.4f}")

        return {"final_loss": float(losses[-1]), "loss_trend": losses}

    def topological_analysis(self):
        data_for_topology = self.processed.reshape(-1, self.config.pca_components).cpu().numpy()

        diagrams = ripser(data_for_topology, maxdim=self.config.topology_maxdim)['dgms']

        plt.figure(figsize=(10, 6))
        plot_diagrams(diagrams, show=False)
        plt.title("Persistence Diagram")
        plt.savefig(os.path.join(self.config.output_dir, "topology.png"))
        plt.close()

        if len(diagrams) > 1:
            H1 = diagrams[1]
            persistence = np.max(H1[:, 1] - H1[:, 0]) if len(H1) > 0 else 0.0
        else:
            persistence = 0.0
        return {"H1_persistence": float(persistence)}

    def visualize_trajectory(self, highlight_steps=None):
        traj = self.processed.cpu().numpy()

        if traj.ndim == 3:
            traj = traj.squeeze(0)

        plt.figure(figsize=(12, 6))

        if self.config.pca_components >= 3:
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
                    if self.config.pca_components >= 3:
                        ax.scatter(traj[step, 0], traj[step, 1], traj[step, 2],
                                   c='red', s=100, marker='x')
                    else:
                        plt.scatter(traj[step, 0], traj[step, 1],
                                    c='red', s=100, marker='x')

        plt.title("State Space Trajectory")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(sc, label="Time Step")

        plt.savefig(os.path.join(self.config.output_dir, f"state_trajectory_seq{self.seq_id}.png"))
        plt.close()

    def run_full_analysis(self):
        results = {}
        continuity = self.analyze_continuity()
        results.update(continuity)

        avg_dist = continuity['mean_distance']
        std_dist = continuity['std_distance']
        jump_threshold = avg_dist + 2 * std_dist
        diffs = torch.diff(self.processed, dim=0).norm(dim=1)
        highlight_steps = torch.where(diffs > jump_threshold)[0].cpu().numpy().tolist()

        results['jump_steps'] = highlight_steps
        self.visualize_trajectory(highlight_steps=highlight_steps)

        self.results = copy.deepcopy(results)

        results.update(self.cluster_analysis())
        results.update(self.temporal_dynamics())
        results.update(self.topological_analysis())
        self.results = results
        return results
