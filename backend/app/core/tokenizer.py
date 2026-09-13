"""真实分词器（票 19）：与生成模型同族的 Qwen 系 tokenizer，惰性加载 + 缓存。

为什么不用字数估算报数：估算能让预算判定跑起来，但**"降了多少"这个数字要能对外讲**，
估算值撑不起这个数字。所以规则是：

- 预算判定：没有真实分词器时回落字符估算（阈值判断不能停）。
- **对外报数：只用真实分词器**；没有就把指标标成「不可用」——绝不静默回退成估算
  （**并且把拿不到的原因写进报告**：只写「未接」等于没说，见 #53）
  （沿用 Spec 0001「缺件不得静默给假数字」）。
"""
from __future__ import annotations

import logging

from app.core.context import ApproxTokenCounter, TokenCounter

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_cache: dict[str, TokenCounter] = {}
_failures: dict[str, str] = {}      # 失败也记：一次拿不到就够，别让每次启动都去打网重试


class HfTokenCounter(TokenCounter):
    """transformers 的 tokenizer 包装。`label` 即模型名 —— 报告里写的就是它。"""

    def __init__(self, tokenizer, model_id: str):
        self._tok = tokenizer
        self.label = model_id

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=False))


def apply_hf_endpoint() -> None:
    """把配置里的 `HF_ENDPOINT` 补进环境变量（已显式设过的优先，不覆盖）。

    huggingface_hub **只认环境变量**，而 `.env` 里的值 pydantic 只灌进 Settings、不进 `os.environ`。
    于是评测进程（不像 run_real.ps1 那样给它显式设过）会直连 huggingface.co —— 国内必失败，
    token 指标整个丢掉，而且报告只印一句笼统的「未接真实分词器」（#53 真机踩到）。
    这一道补上，`.env` 才真的「一处配置，处处生效」。
    """
    import os

    from app.config import get_settings

    endpoint = get_settings().hf_endpoint
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)


def _hf_loader(model_id: str):
    """默认加载器：惰性 import transformers，从 HF（或其镜像）取 tokenizer 文件。"""
    apply_hf_endpoint()          # 镜像从**配置**来 —— 不然国内直连 hf 必失败
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def try_real_token_counter(model_id: str = DEFAULT_MODEL, *, loader=None) -> tuple[TokenCounter | None, str]:
    """尝试拿到真实分词器。返回 (计数器, 说明)；拿不到时计数器为 None、说明写原因。

    **不算失败**：没装 transformers / 下不到模型 / 模型名不对 —— 都只是「这个指标不可用」，
    调用方据此把 token 指标标成不可用（而不是编一个估算值出来）。
    """
    loader = loader or _hf_loader
    if not model_id:
        # 没配就是没配 —— 不去猜、不去下载，指标直接记为不可用
        return None, "未配置真实分词器（TOKENIZER_MODEL 为空）"
    if model_id in _cache:
        return _cache[model_id], "缓存命中"
    if model_id in _failures:
        return None, _failures[model_id]
    try:
        counter = HfTokenCounter(loader(model_id), model_id)
    except Exception as e:      # noqa: BLE001 —— 凡是拿不到就都算「不可用」，原因如实写出来
        note = "真实分词器 %s 不可用：%s: %s" % (model_id, type(e).__name__, e)
        logger.warning("%s（token 指标将标记为不可用）", note)
        _failures[model_id] = note
        return None, note
    _cache[model_id] = counter
    logger.info("真实分词器已就绪：%s", model_id)
    return counter, ""


def setup_token_counter(model_id: str = DEFAULT_MODEL, *, loader=None) -> tuple[TokenCounter, str]:
    """预算用计数器 + 报数口径说明。返回 (计数器, note)。

    拿不到真实分词器时返回**字符估算计数器**（预算仍能跑），note 里写明原因 ——
    调用方报数前必须检查 `counter.label`，为空就不许报 token 数字。
    """
    real, note = try_real_token_counter(model_id, loader=loader)
    if real is not None:
        return real, ""
    approx = ApproxTokenCounter()
    approx.note = note          # 实例属性：报告「不可用」那行会原样带上原因
    return approx, note
