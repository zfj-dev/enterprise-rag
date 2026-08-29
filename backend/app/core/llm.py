"""LLM 抽象：`stream(messages)` 产出增量文本。含 Fake（离线 demo/测试）与云端 API（OpenAI 兼容）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from app.config import get_settings

FAKE_ANSWER = (
    "（模拟回答）根据检索到的资料，这是一种基于检索增强生成（RAG）的问答：系统先对您的文档做解析、分块、向量化，"
    "提问时混合检索 + 重排召回相关片段，再交给大模型结合原文作答，并附上来源引用。本回答为离线演示（未接云端 LLM）。[来源: 演示文库]"
)


class LLM(ABC):
    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        ...


class FakeLLM(LLM):
    def stream(self, messages: list[dict]) -> Iterator[str]:
        for piece in _chunk_text(FAKE_ANSWER, 20):
            yield piece


class CloudLLM(LLM):
    """OpenAI 兼容接口（DeepSeek / SiliconFlow / qwen），text-event-stream 增量。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None):
        import httpx

        s = get_settings()
        self._httpx = httpx
        self.base_url = (base_url or s.all_llm_url()).rstrip("/")
        self.api_key = api_key or s.llm_api_key or ""
        self.model = model or s.llm_model
        self.temperature = temperature if temperature is not None else s.llm_temperature
        self.max_tokens = max_tokens or s.llm_max_tokens

    def stream(self, messages: list[dict]) -> Iterator[str]:
        if not self.api_key:
            yield "服务配置缺失（未设置 LLM API Key）。"
            return
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens, "stream": True}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with self._httpx.Client(timeout=60) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except Exception:
                        continue
                    if delta:
                        yield delta


def get_llm(provider: str | None = None) -> LLM:
    provider = provider or get_settings().llm_provider
    if provider in ("deepseek", "siliconflow", "openai", "dashscope"):
        return CloudLLM()
    return FakeLLM()


def _chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]
