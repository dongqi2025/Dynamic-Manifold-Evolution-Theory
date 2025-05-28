from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return tokenizer, model.eval()
