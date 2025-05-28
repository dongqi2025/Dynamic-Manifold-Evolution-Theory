import pandas as pd
import numpy as np
import logging
from statsmodels.regression.mixed_linear_model import MixedLM


def statistical_validation(experiment_reports, config):
    data = []
    for exp_id, report in enumerate(experiment_reports):
        num_texts = len(report['quality']['perplexity'])
        for text_idx in range(num_texts):
            dyn = report['dynamics'][text_idx]
            qual = report['quality']
            data.append({
                'exp_id': exp_id,
                'continuity': dyn['mean_distance'],
                'cluster_quality': dyn['silhouette_score'],
                'predictability': 1 / (dyn['final_loss'] + 1e-6),
                'topology': dyn['H1_persistence'],
                'log_ppl': np.log(qual['perplexity'][text_idx]),
                'spelling': qual['spelling'][text_idx],
                'diversity': qual['diversity'][text_idx],
                'grammar': qual['grammar'][text_idx],
                'coherence': qual['coherence'][text_idx]
            })

    df = pd.DataFrame(data)

    results = {}
    for target in ['log_ppl', 'spelling', 'diversity', 'grammar', 'coherence']:
        if len(df) < config.statistical_thresholds['min_sample_size']:
            logging.warning("样本量不足，跳过混合效应模型")
            continue

        try:
            model = MixedLM.from_formula(
                f"{target} ~ continuity + cluster_quality + predictability + topology",
                groups=df['exp_id'],
                data=df
            ).fit()

            results[target] = {
                'params': model.params.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'aic': model.aic
            }
        except Exception as e:
            logging.error(f"建模失败: {str(e)}")
            results[target] = {"error": str(e)}

    return results
