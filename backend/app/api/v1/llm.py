"""自带 Key（BYOK）：填入 / 查看 / 轮换 / 删除自己的 LLM 配置（票 32）。

**明文 key 只进来一次，之后一步都不出去**：读接口只回尾号；`LLMConfig` 的 repr 里没有 key，
所以日志与 trace 也带不出它。删掉即立刻回落服务端全局。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_runtime
from app.config import get_settings
from app.core.byok import LLMConfig
from app.core.container import Runtime
from app.core.ssrf import check_base_url, parse_allowed_hosts
from app.core.schemas import LLMConfigIn, LLMConfigOut
from app.models.entities import User

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/config", response_model=LLMConfigOut)
def get_config(user: User = Depends(get_current_user),
               rt: Runtime = Depends(get_runtime)) -> LLMConfigOut:
    """看自己配了什么 —— **只有 base_url / model / 尾号**。"""
    return _out(rt, user.id)


@router.put("/config", response_model=LLMConfigOut)
def put_config(body: LLMConfigIn, user: User = Depends(get_current_user),
               rt: Runtime = Depends(get_runtime)) -> LLMConfigOut:
    """填入或**轮换**：直接覆盖，旧凭据随即失效。

    先过 **SSRF 防护**（票 33）：私网 / 回环 / 非 https 一律拒，拒绝原因写清楚是地址不合规 ——
    而不是等真去连的时候报一个含糊的连接错误。
    """
    s = get_settings()
    reason = check_base_url(body.base_url, allowed_hosts=parse_allowed_hosts(s.byok_allowed_hosts),
                            allow_insecure=s.byok_allow_insecure, resolve=rt.url_resolver)
    if reason:
        from fastapi import HTTPException
        raise HTTPException(400, reason)
    rt.user_llm_config_store.set(user.id, LLMConfig(base_url=body.base_url.strip(),
                                                    api_key=body.key, model=body.model.strip()))
    return _out(rt, user.id)


@router.delete("/config")
def delete_config(user: User = Depends(get_current_user), rt: Runtime = Depends(get_runtime)):
    """删掉自己的配置 —— 下一次问答立刻回落服务端全局。"""
    rt.user_llm_config_store.delete(user.id)
    return {"ok": True}


def _out(rt: Runtime, user_id: str) -> LLMConfigOut:
    view = rt.user_llm_config_store.public_view(user_id)
    # persistent 如实告诉用户：没配加密口令时凭据只在内存里，重启就没了
    return LLMConfigOut(configured=bool(view), persistent=rt.user_llm_config_store.persistent,
                        **(view or {}))
