# Dynamic-Manifold-Evolution-Theory

## 📂 Repository Structure
```text
project_root/
│
├── config/
│   └── experiment_config.py  # 实验配置类的定义
├── data_processing/
│   ├── data_preprocessor.py  # 数据预处理相关代码
│   └── hidden_state_collector.py  # 隐藏状态收集代码
├── analysis/
│   ├── dynamical_analyzer.py  # 动态分析类及相关方法
│   ├── correlation_analyzer.py  # 相关性分析代码
│   ├── statistical_validator.py  # 统计验证代码
│   └── text_evaluator.py  # 文本评估代码
├── visualization/
│   ├── trajectory_visualizer.py  # 轨迹可视化代码
│   ├── analysis_visualizer.py  # 分析结果可视化代码
│   └── gif_processor.py  # GIF处理代码
├── models/
│   └── model_loader.py  # 模型加载代码
├── main.py  # 主程序入口，运行实验和多实验分析

```

## Quick start

1. 安装依赖：pip install -r requirements.txt
2. 配置参数：在config/experiment_config.py文件中，可以根据需要修改实验配置参数。 
3. 运行代码：在项目根目录下，执行以下命令来运行多实验分析：python main.py
