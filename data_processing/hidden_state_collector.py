def collect_hidden_states(outputs):
    """
    优化后的GPU端隐藏状态采集
    返回形状: (num_sequences, seq_len, hidden_dim)
    """
    assert hasattr(outputs, "hidden_states"), "需要确保生成时output_hidden_states=True"

    # 保持所有计算在GPU
    layer_states = [
        step[-1].to(torch.float32)  # 转换到float32避免精度问题
        for step in outputs.hidden_states[1:]  # 跳过初始输入状态
    ]

    # 使用PyTorch进行维度重组
    stacked = torch.cat(layer_states, dim=1)  # (batch*num, seq_len, dim)
    return stacked.reshape(-1, *stacked.shape[1:])  # (num_sequences, seq_len, dim)
