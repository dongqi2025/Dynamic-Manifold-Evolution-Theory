import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def visualize_analysis(results, config):
    continuity = [d['mean_distance'] for d in results['dynamics']]
    cluster_quality = [d['silhouette_score'] for d in results['dynamics']]
    predictability = [1 / (d['final_loss'] + 1e-6) for d in results['dynamics']]
    topology = [d['H1_persistence'] for d in results['dynamics']]

    df = pd.DataFrame({
        'Continuity': continuity,
        'Cluster Quality': cluster_quality,
        'Predictability': predictability,
        'Topology': topology,
        'Log Perplexity': np.log(results['quality']['perplexity']),
        'Spelling': results['quality']['spelling'],
        'Diversity': results['quality']['diversity'],
        'Grammar': results['quality']['grammar'],
        'Coherence': results['quality']['coherence']
    })

    plt.figure(figsize=(15, 10))
    sns.pairplot(df,
                 x_vars=['Continuity', 'Cluster Quality', 'Predictability', 'Topology'],
                 y_vars=['Log Perplexity', 'Spelling', 'Diversity', 'Grammar', 'Coherence'],
                 kind='reg',
                 plot_kws={'scatter_kws': {'alpha': 0.5}})
    plt.savefig(os.path.join(config.output_dir, "full_correlation_matrix.png"))
    plt.close()
