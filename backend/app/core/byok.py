"""BYOK（自带 Key）：用户配置 → LLM 实例。

本模块只做「按请求解析 LLM」这条链路：**工厂与配置来源都从外部注入**，
默认配置来源为空（谁都没配过），因此未配置时一律回落到服务端全局 LLM，行为与今天完全一致。

key 的加密落库、不回显纪律、SSRF 防护属于后续票据（32/33），不在此实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.llm import CloudLLM, LLM


@dataclass(frozen=True)
class LLMConfig:
    """一份自带 LLM 配置。一律按 OpenAI 兼容协议调用，不为各家新写适配器。"""

    base_url: str
    api_key: str
    model: str


class UserLLMConfigStore(ABC):
    """按用户存取自带配置。默认无配置 → 全部回落服务端全局。"""

    @abstractmethod
    def get(self, user_id: str) -> LLMConfig | None: ...

    @abstractmethod
    def set(self, user_id: str, cfg: LLMConfig) -> None: ...

    @abstractmethod
    def delete(self, user_id: str) -> None: ...


class InMemoryUserLLMConfigStore(UserLLMConfigStore):
    """内存实现（测试与单进程运行用）。加密落库版见后续票据。"""

    def __init__(self) -> None:
        self._by_user: dict[str, LLMConfig] = {}

    def get(self, user_id: str) -> LLMConfig | None:
        return self._by_user.get(user_id)

    def set(self, user_id: str, cfg: LLMConfig) -> None:
        self._by_user[user_id] = cfg

    def delete(self, user_id: str) -> None:
        self._by_user.pop(user_id, None)


class LLMFactory(ABC):
    """把一份用户配置构造成 LLM 实例（从外部注入，便于用 stub 做确定性测试）。"""

    @abstractmethod
    def build(self, cfg: LLMConfig) -> LLM: ...


class OpenAICompatLLMFactory(LLMFactory):
    """默认工厂：按 OpenAI 兼容协议构造（覆盖 DeepSeek / SiliconFlow / DashScope / OpenAI 等）。"""

    def build(self, cfg: LLMConfig) -> LLM:
        return CloudLLM(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model)
