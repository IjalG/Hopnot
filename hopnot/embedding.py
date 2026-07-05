"""嵌入接口 —— 语义向量生成抽象层。

提供 BaseEmbedding 抽象基类，以及一个基于随机向量的 DummyEmbedding
（用于测试/开发）。实际使用时接入冻结的千问 0.5B 或任何文本嵌入模型。
"""

from __future__ import annotations

import hashlib
import math
import random
from abc import ABC, abstractmethod
from typing import Optional


class BaseEmbedding(ABC):
    """语义嵌入抽象基类。

    所有向量均为 L2 归一化后的单位向量。
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """将文本转换为 L2 归一化的语义向量。"""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。"""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入，默认逐个调用 embed。"""
        return [self.embed(t) for t in texts]


class Qwen3Embedding(BaseEmbedding):
    """基于 Qwen3-Embedding-0.6B 的专用语义嵌入模型（推荐）。

    从 ModelScope 自动加载，支持两种模型：
    1. Qwen3-Embedding-0.6B（专用嵌入模型，推荐）：sentence-transformers 格式
    2. Qwen3-0.6B（通用语言模型，备用）：last-token pooling + L2 归一化

    所有权重冻结，带嵌入缓存。

    Usage:
        embedder = Qwen3Embedding(device="cpu")
        vec = embedder.embed("人工智能")
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        max_length: int = 128,
    ) -> None:
        import os

        if not model_path:
            cache_base = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                ".model_cache", "Qwen",
            )
            embedding_path = os.path.join(cache_base, "Qwen3-Embedding-0___6B")
            plain_path = os.path.join(cache_base, "Qwen3-0___6B")

            if os.path.isdir(embedding_path):
                model_path = embedding_path
            elif os.path.isdir(plain_path):
                model_path = plain_path
            else:
                # 都没缓存 → 从 ModelScope 自动下载专用嵌入模型
                print("正在从 ModelScope 下载 Qwen3-Embedding-0.6B...", flush=True)
                try:
                    from modelscope import snapshot_download
                    snapshot_download("Qwen/Qwen3-Embedding-0.6B", cache_dir=os.path.dirname(cache_base))
                    # 下载后路径
                    model_path = os.path.join(cache_base, "Qwen3-Embedding-0___6B")
                except Exception as e:
                    raise RuntimeError(f"ModelScope 下载失败: {e}。请检查网络或用 model_path 指定本地路径。")

        self._device = device
        self._max_length = max_length
        self._cache: dict[str, list[float]] = {}
        self._using_st = False

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"模型路径不存在: {model_path}\n"
                f"请确保模型已下载，或删除缓存目录让系统自动下载。"
            )

        is_st_format = (
            os.path.isdir(os.path.join(model_path, "1_Pooling"))
            and os.path.isfile(os.path.join(model_path, "modules.json"))
        )

        if is_st_format:
            self._init_with_st(model_path, device)
        else:
            self._init_with_hf(model_path, device)

    def _init_with_st(self, model_path: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_path, device=device)
        self._dim = self._model.get_embedding_dimension()  # sentence-transformers 4.x+
        self._using_st = True

    def _init_with_hf(self, model_path: str, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float32,
        ).to(device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False
        self._dim = self._model.config.hidden_size

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text].copy()
        if self._using_st:
            vec = self._model.encode(text, normalize_embeddings=True)
            vec = [float(x) for x in vec]
        else:
            import torch
            prefixed = f'为这个句子生成向量：{text}'
            inputs = self._tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=self._max_length).to(self._device)
            with torch.no_grad():
                vec = self._model(**inputs).last_hidden_state[0, -1, :].cpu().numpy()
            norm = float(torch.tensor(vec).norm().item())
            vec = [float(x / norm) for x in vec] if norm > 0 else [0.0] * self._dim
        self._cache[text] = vec
        return vec.copy()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            if self._using_st:
                vecs = self._model.encode(uncached, normalize_embeddings=True)
                for text, vec in zip(uncached, vecs):
                    self._cache[text] = [float(x) for x in vec]
            else:
                import torch
                prefixed = [f'为这个句子生成向量：{t}' for t in uncached]
                inputs = self._tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True, max_length=self._max_length).to(self._device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    mask = inputs.get("attention_mask")
                    if mask is not None:
                        seq_lens = mask.sum(dim=1) - 1
                        pooled = outputs.last_hidden_state[torch.arange(len(uncached), device=self._device), seq_lens]
                    else:
                        pooled = outputs.last_hidden_state[:, -1, :]
                for i, text in enumerate(uncached):
                    vec = pooled[i].cpu().numpy()
                    norm = float(torch.tensor(vec).norm().item())
                    self._cache[text] = [float(x / norm) for x in vec] if norm > 0 else [0.0] * self._dim
        return [self._cache[t].copy() for t in texts]

    def clear_cache(self) -> None:
        """清空嵌入缓存。"""
        self._cache.clear()


class DummyEmbedding(BaseEmbedding):
    """基于确定性的伪随机嵌入（用于开发/测试）。

    相同文本返回相同向量，不同文本返回不同向量。
    """

    def __init__(self, dim: int = 64, seed: int = 42) -> None:
        self._dim = dim
        self._seed = seed
        self._cache: dict[str, list[float]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text].copy()

        # 使用 hashlib 确定性地生成向量（避免 Python hash() 的进程间随机化）
        h_bytes = (text + str(self._seed)).encode('utf-8')
        h = int(hashlib.md5(h_bytes).hexdigest(), 16)
        rng = random.Random(h)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self._dim)]
        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        self._cache[text] = vec
        return vec.copy()
