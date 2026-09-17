"""LLM 抽象：`stream(messages, usage=None)` 产出增量文本。含 Fake（离线 demo/测试）与云端 API（OpenAI 兼容）。

另有 `chat_with_tools(messages, tools, usage=None)` —— 带工具的一轮对话（返回内容 + 工具调用列表）。
**基类给了会降级的默认实现**：不支持工具的模型就当成普通一问一答，依赖方自然退化为单步回答。

`usage` 见 `put_usage`：要记账的调用方每次传一个**自己的**空 dict，这次调用的账单口径写在那里。
以前它挂在实例上（`last_usage`），而实例是进程级共享的 —— 并发请求、以及生成之后紧跟着的
引用校验与事实抽取会互相覆盖，用量/费用就记到别人头上了（#64 批 3）。
"""
from __future__ import annotations

import json
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """模型要求调用的一次工具。`arguments` 已解析成 dict；解析不了就是空 dict（原文留在 raw）。"""

    id: str
    name: str
    arguments: dict = field(default_factory=dict)
    raw: str = ""


def encode_tool_calls(calls) -> list[dict]:
    """ToolCall 列表 -> OpenAI 兼容的 `tool_calls`（回灌给模型时的形状）。

    与 parse_tool_calls 成对放在一起 —— 同一份协议只该有一个地方知道它长什么样。
    """
    return [{"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
            for c in calls]


def parse_tool_calls(raw) -> list["ToolCall"]:
    """把 OpenAI 兼容协议里的 `tool_calls` 解析成 ToolCall 列表。

    参数里的 JSON 可能是坏的（模型偶尔吐不合法 JSON、或把 JSON 裹进 ``` 代码块），
    解析不了就**留空 dict + 原文**，让上层自己决定怎么办 —— **这里不抛**（元素不是 dict 也不抛）。

    约定：参数不是 JSON 对象时（比如模型直接给了个数字），塞进 `{"value": ...}`。
    """
    out = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue                      # 协议外的杂项直接跳过，别让它把整轮对话带崩
        fn = item.get("function") if isinstance(item.get("function"), dict) else {}
        text = str(fn.get("arguments") or "")
        args: dict = {}
        if text:
            parsed = extract_json(text, "{")    # 先在文本里抠 JSON 对象（兼容裹代码块的写法）
            if isinstance(parsed, dict):
                args = parsed
            else:
                try:
                    value = json.loads(text)
                    args = value if isinstance(value, dict) else {"value": value}
                except Exception:
                    args = {}
        out.append(ToolCall(id=str(item.get("id") or ""), name=str(fn.get("name") or ""),
                            arguments=args, raw=text))
    return out

import time

from app.config import get_settings
from app.utils.text import approx_token_count, extract_json

FAKE_ANSWER = (
    "（模拟回答）根据检索到的资料，这是一种基于检索增强生成（RAG）的问答：系统先对您的文档做解析、分块、向量化，"
    "提问时混合检索 + 重排召回相关片段，再交给大模型结合原文作答，并附上来源引用。本回答为离线演示（未接云端 LLM）。[来源: 演示文库]"
)


def put_usage(sink: dict | None, data) -> None:
    """把**这一次调用**的账单口径写进调用方自己的桶（票 27 / #64 批 3）。

    口径是该次调用的产出，不能挂在模型实例上：实例是进程级共享的，并发的请求（以及紧接着的
    引用校验、事实抽取）会把它覆盖掉，于是这次问答记了别人的账。没给桶就是调用方不需要这个
    数，直接不记。`data` 为空也不记 —— 空桶本身已经表示「这次拿不到」。
    """
    if sink is None or not data:
        return
    sink.clear()
    sink.update(data)


class LLM(ABC):
    is_fake: bool = False   # 演示/测试用的假模型；真实模型为 False

    @abstractmethod
    def stream(self, messages: list[dict], usage: dict | None = None) -> Iterator[str]:
        """产出增量文本。`usage` 见 `put_usage` —— 要记账的调用方自己传一个空 dict 进来。"""

    def chat_with_tools(self, messages: list[dict],
                        tools: Sequence[dict] | None = None,
                        usage: dict | None = None) -> dict:
        """带工具的一轮对话：返回 {"content": str, "tool_calls": [ToolCall, ...]}。

        **默认实现明确降级**：把消息当普通一问一答，工具调用为空 —— 不支持工具的模型
        （含演示用的 Fake）不会抛穿，依赖方自然退化成单步回答。
        """
        return {"content": "".join(self.stream(messages, usage=usage)), "tool_calls": []}


class FakeLLM(LLM):
    is_fake = True

    def stream(self, messages: list[dict], usage: dict | None = None) -> Iterator[str]:
        # 演示用假模型**自报**一份「模拟用量」（票 30 要求 demo 下也能演示计量链路）——
        # 打的标记是 simulated，记账那边据此标成**模拟口径**，绝不冒充 provider 账单。
        prompt = "".join(str(m.get("content") or "") for m in messages)
        put_usage(usage, {"prompt_tokens": approx_token_count(prompt),
                          "completion_tokens": approx_token_count(FAKE_ANSWER),
                          "simulated": True})
        delay = get_settings().fake_llm_delay
        for piece in _chunk_text(FAKE_ANSWER, 20):
            if delay:
                time.sleep(delay)
            yield piece


class CloudLLM(LLM):
    """OpenAI 兼容接口（DeepSeek / SiliconFlow / qwen），text-event-stream 增量。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None, timeout: float | None = None):
        import httpx

        s = get_settings()
        self._httpx = httpx
        self.timeout = 60.0 if timeout is None else float(timeout)   # 0 不静默变 60
        self.base_url = (base_url or s.all_llm_url()).rstrip("/")
        self.api_key = api_key or s.llm_api_key or ""
        self.model = model or s.llm_model
        self.temperature = temperature if temperature is not None else s.llm_temperature
        self.max_tokens = max_tokens or s.llm_max_tokens

    def _post_payload(self, messages: list[dict], tools=None, stream: bool = True) -> dict:
        """请求体只在这里拼一份 —— 流式与带工具都走它，免得两边各写一遍还写不一致。"""
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens, "stream": stream}
        if tools:
            payload["tools"] = list(tools)
        return payload

    def _stream_lines(self, url: str, payload: dict, headers: dict,
                      usage: dict | None = None) -> Iterator[str]:
        """跑一次流式请求，逐 chunk 解析（顺带把 usage 写进调用方的桶）。

        `timeout` 封的是**连接**与**两次数据之间的等待**（httpx 的 read 是「等下一个 chunk」），
        **不封整段生成的墙钟时间** —— 一个长回答本来就该慢慢流完，掐掉它才是错的。
        """
        with self._httpx.Client(timeout=self.timeout) as client:
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
                        chunk = json.loads(data)
                    except Exception:
                        continue
                    # 带 usage 的那个 chunk 通常 choices 为空 —— 先取 usage，再看有没有正文
                    if chunk.get("usage"):
                        put_usage(usage, chunk["usage"])
                    choices = chunk.get("choices") or []
                    delta = choices[0].get("delta", {}).get("content") if choices else None
                    if delta:
                        yield delta

    def stream(self, messages: list[dict], usage: dict | None = None) -> Iterator[str]:
        # 不需要「先清掉上一次」了：桶是调用方按次给的，天生就是空的
        if not self.api_key:
            yield "服务配置缺失（未设置 LLM API Key）。"
            return
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._post_payload(messages)
        # 让 OpenAI 兼容的 provider 在最后一个 chunk 里带上 usage（与账单同口径）
        payload["stream_options"] = {"include_usage": True}
        try:
            yield from self._stream_lines(url, payload, headers, usage)
            return
        except self._httpx.HTTPStatusError as e:
            if not (400 <= e.response.status_code < 500):
                raise
            # provider 不认这个字段 —— 去掉它重来一次（这一次拿不到 usage，记账回退本地口径）。
            # 状态码是在吐第一个字之前就检查的，所以重试不会把正文吐两遍。
            logger.warning("provider 拒绝了 stream_options（%s），去掉后重试：本次拿不到账单口径", e)
        payload.pop("stream_options", None)
        yield from self._stream_lines(url, payload, headers, usage)

    def chat_with_tools(self, messages: list[dict],
                        tools: Sequence[dict] | None = None,
                        usage: dict | None = None) -> dict:
        """带工具的一轮（非流式）。没配 Key / 没给工具 / 调用失败，都**降级**成一问一答。

        降级后若连普通回答也失败（provider 整个不可用），异常照常上抛 ——
        那是模型本身挂了，不是「不支持工具」，不该在这里被吞掉。
        """
        if not tools or not self.api_key:
            return super().chat_with_tools(messages, tools, usage)

        url = f"{self.base_url}/chat/completions"
        payload = self._post_payload(messages, tools=tools, stream=False)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            with self._httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()
                message = body["choices"][0]["message"]
        except Exception as e:   # noqa: BLE001 —— 不支持工具的 provider 会在这里炸，降级而不是抛穿
            logger.warning("带工具对话失败，降级为普通回答：%s", e)
            return super().chat_with_tools(messages, tools, usage)

        put_usage(usage, body.get("usage"))      # 非流式响应里 usage 在顶层（与账单同口径）
        return {"content": message.get("content") or "",
                "tool_calls": parse_tool_calls(message.get("tool_calls"))}


def get_llm(provider: str | None = None) -> LLM:
    provider = provider or get_settings().llm_provider
    if provider in ("deepseek", "siliconflow", "openai", "dashscope"):
        return CloudLLM()
    return FakeLLM()


def _chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]
