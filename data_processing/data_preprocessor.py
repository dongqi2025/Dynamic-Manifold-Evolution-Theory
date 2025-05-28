import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.pca_components)
        self.device = "cuda"

    def preprocess(self, hidden_states):
        assert hidden_states.ndim == 3, f"输入维度错误，应为(batch, seq_len, dim)，实际得到{hidden_states.shape}"
        if hidden_states.size(0) == 1:
            states = hidden_states.squeeze(0).float()  # (seq_len, dim)
        else:
            states = hidden_states.float()  # (batch, seq_len, dim)

        if states.ndim == 3:
            batch_size, seq_len, feat_dim = states.shape
            states_np = states.cpu().numpy().astype(np.float32)
            reshaped = states_np.reshape(-1, feat_dim)
        else:
            seq_len, feat_dim = states.shape
            states_np = states.cpu().numpy().astype(np.float32)
            reshaped = states_np.reshape(-1, feat_dim)

        scaled = self.scaler.fit_transform(reshaped)
        pca_proj = self.pca.fit_transform(scaled)

        if states.ndim == 3:
            return torch.from_numpy(
                pca_proj.reshape(batch_size, seq_len, -1)
            ).to(self.device)
        else:
            return torch.from_numpy(
                pca_proj.reshape(seq_len, -1)
            ).to(self.device)