import numpy as np
import torch
from torch import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, T5EncoderModel
from spellchecker import SpellChecker
from collections import defaultdict
from typing import List, Dict, Tuple
import re


class TextEvaluator:
    def __init__(self,
                 gpt2_path: str = "/ossfs/workspace/models/openai-community__gpt2",
                 t5_path: str = "/ossfs/workspace/models/sentence-transformers__sentence-t5-xxl",
                 device: str = "cuda"):
        """
        完全离线的文本质量评估器

        参数:
            gpt2_path: GPT-2模型本地路径
            t5_path:sentence-t5-xxl模型本地路径
            device: 模型运行设备 (cuda/cpu)
        """
        self.device = device

        # 初始化拼写检查器
        self.spell_checker = SpellChecker()

        # 加载本地GPT-2模型
        print("loading gpt2...")
        self.tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
        self.model = AutoModelForCausalLM.from_pretrained(gpt2_path).to(device)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载 T5 模型
        print("loading t5...")
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_path)
        self.t5_model = T5EncoderModel.from_pretrained(t5_path).to(device)

        # 初始化统计参数
        self.ngram_counts = defaultdict(int)
        self.total_words = 0

    def evaluate(self, texts: List[str]) -> Dict[str, List[float]]:
        """
        执行完整评估流程

        返回:
            metrics: 包含各项指标的字典
            - perplexity: 困惑度 (越低越好)
            - spelling: 拼写正确率 (0-1)
            - diversity: 二元语法多样性 (0-1)
            - grammar: 简单语法正确率 (0-1)
            - coherence: 语义一致性得分 (0-1)

        """
        return {
            "perplexity": self._calc_perplexity(texts),
            "spelling": self._spelling_check(texts),
            "diversity": [self._ngram_diversity(text) for text in texts],
            "grammar": [self._rule_based_grammar(text) for text in texts],
            "coherence": self._semantic_coherence(texts)
        }

    def _calc_perplexity(self, texts: List[str]) -> List[float]:
        """计算基于GPT-2的困惑度"""
        print("_calc_perplexity...")
        ppl_scores = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                ppl = torch.exp(loss).cpu().item()

            ppl_scores.append(ppl)
        return ppl_scores

    def _spelling_check(self, texts: List[str]) -> List[float]:
        """离线拼写检查"""
        print("_spelling_check...")
        scores = []
        for text in texts:
            words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
            if not words:
                scores.append(0.0)
                continue

            misspelled = self.spell_checker.unknown(words)
            scores.append(1 - len(misspelled) / len(words))
        return scores

    def _ngram_diversity(self, text: str, n: int = 2) -> float:
        """计算n-gram多样性"""
        print("_spelling_check...")
        words = re.findall(r'\b\w+\b', text.lower())
        total = max(len(words) - n + 1, 1)

        unique_ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])
            unique_ngrams.add(ngram)
            self.ngram_counts[ngram] += 1
            self.total_words += 1

        return len(unique_ngrams) / total

    def _rule_based_grammar(self, text: str) -> float:
        """基于规则的简单语法检查"""
        print("_rule_based_grammar...")
        error_count = 0
        sentences = re.split(r'[.!?]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 检查句首大写
            if sentence[0].islower():
                error_count += 1

            # 检查基本的主谓结构（简单版）
            words = sentence.split()
            if len(words) >= 2:
                if words[0].endswith('s') and words[1].startswith('ing'):
                    error_count += 1

        return 1 - error_count / (len(sentences) + 1e-6)

    def _semantic_coherence(self, texts: List[str]) -> List[float]:
        """使用 T5 的语义一致性评估"""
        print("_semantic_coherence...")
        scores = []
        for text in texts:
            sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

            if len(sentences) < 2:
                scores.append(0.5)
                continue

            # T5 需要添加前缀
            inputs = self.t5_tokenizer(
                [f"sentence: {s}" for s in sentences],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.t5_model(**inputs)
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs.attention_mask)

            # 计算相邻句子对的相似度
            if embeddings.shape[0] > 1:
                # 前n-1个句子和後n-1个句子配对
                prev_emb = embeddings[:-1]
                next_emb = embeddings[1:]
                similarities = cosine_similarity(prev_emb, next_emb, dim=1)
                scores.append(similarities.mean().item())
            else:
                scores.append(0.5)

        return scores

    def _mean_pooling(self, token_embeddings, attention_mask):
        """T5 需要特殊池化处理"""
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_vocab_stats(self) -> Tuple[float, int]:
        """获取全局词汇统计信息"""
        unique_ngrams = len(self.ngram_counts)
        return (unique_ngrams / self.total_words if self.total_words > 0 else 0,
                unique_ngrams)

