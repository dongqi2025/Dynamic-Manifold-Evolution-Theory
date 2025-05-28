import os
import json
import logging
import copy
from config.experiment_config import ExperimentConfig
from models.model_loader import load_model
from data_processing.hidden_state_collector import collect_hidden_states
from analysis.dynamical_analyzer import DynamicalAnalyzer
from analysis.correlation_analyzer import analyze_correlation
from analysis.text_evaluator import TextEvaluator
from analysis.statistical_validator import statistical_validation
from visualization.analysis_visualizer import visualize_analysis
from visualization.gif_processor import process_gif


def run_experiment(config):
    os.makedirs(config.output_dir, exist_ok=True)

    try:
        tokenizer, model = load_model(config)

        inputs = tokenizer(config.prompt, return_tensors="pt").to(model.device)
        gen_args = {
            "max_length": config.max_length,
            "num_return_sequences": config.num_return_sequences,
            "temperature": config.generation_temp,
            "top_p": config.top_p,
            "output_hidden_states": True,
            "return_dict_in_generate": True
        }

        outputs = model.generate(**inputs, **gen_args)
        states = collect_hidden_states(outputs)
        np.save(os.path.join(config.output_dir, "all_states.npy"),
                states.cpu().numpy())

        assert states.shape[0] == config.num_return_sequences, \
            f"状态维度{states.shape}与生成数量{config.num_return_sequences}不匹配"
        texts = [tokenizer.decode(seq, skip_special_tokens=True)
                 for seq in outputs.sequences]

        dynamics_results = []
        for i in range(config.num_return_sequences):
            sequence_states = states[i:i + 1]
            analyzer = DynamicalAnalyzer(config, sequence_states, seq_id=i)
            dynamics_results.append(analyzer.run_full_analysis())

            sequence_text = tokenizer.decode(outputs.sequences[i],
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
            sequence_tokens = sequence_text.split()
            # 这里需要实现 generate_evolution_animation 函数
            # generate_evolution_animation(analyzer, sequence_tokens, config, seq_id=i)

        quality_results = TextEvaluator().evaluate(texts)

        analysis_result = analyze_correlation(dynamics_results, quality_results)

        final_report = {
            "dynamics": dynamics_results,
            "quality": quality_results,
            "correlation_analysis": analysis_result,
            "config": vars(config),
            "gen_args": gen_args,
            "texts": texts
        }
        with open(os.path.join(config.output_dir, "report.json"), "w") as f:
            json.dump(final_report, f, indent=2)
        visualize_analysis(final_report, config)
        logging.info(f"Experiment results saved to {config.output_dir}")

    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise


def run_multiple_experiments():
    base_config = ExperimentConfig()
    reports = []
    output_dirs = []

    for temp in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]:
        for top_p in [0.3, 0.6, 0.8, 1.0]:
            exp_prefix = f"temp{temp}_topP{top_p}"

            output_root = Path(base_config.root_output_dir)
            existing_exps = [
                d for d in output_root.iterdir()
                if d.is_dir() and d.name.startswith(exp_prefix)
            ]

            if existing_exps:
                logging.info(f"实验 {exp_prefix} 已存在，跳过执行")
                output_dirs.extend([str(d) for d in existing_exps])
                continue

            config = copy.deepcopy(base_config)
            config.generation_temp = temp
            config.top_p = top_p

            config.experiment_name = exp_prefix
            run_experiment(config)
            output_dirs.append(config.output_dir)

    for path in output_dirs:
        with open(os.path.join(path, "report.json")) as f:
            reports.append(json.load(f))

    stats = statistical_validation(reports, base_config)

    with open(f"{base_config.root_output_dir}/meta_analysis.json", "w") as f:
        json.dump(stats, f, indent=2)

    # run_cross_experiment_analysis(base_config)


if __name__ == "__main__":
    run_multiple_experiments()
