import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=False)
class ExperimentConfig:
    root_output_dir: str = "results"
    experiment_name: str = "deepseek_analysis"
    model_path: str = "../models/DeepSeek-R1-Distill-Qwen-7B"
    prompt: str = "The future of AI is"
    max_length: int = 100
    num_return_sequences: int = 10
    pca_components: int = 2
    cluster_n: int = 3
    topology_maxdim: int = 1
    lstm_hidden_dim: int = 256
    lstm_epochs: int = 100
    generation_temp: float = 1.0
    top_p: float = 0.9
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # 验证阈值
    continuity_mean_upper: float = 0.5
    continuity_std_upper: float = 0.3
    topology_persistence: float = 0.2
    silhouette_threshold: float = 0.4
    acf_lag5_threshold: float = 0.3

    statistical_thresholds: dict = field(default_factory=lambda: {
        "correlation_significance": 0.05,
        "min_sample_size": 30
    })

    @property
    def output_dir(self):
        return os.path.join(self.root_output_dir, f"{self.experiment_name}_{self.timestamp}")
