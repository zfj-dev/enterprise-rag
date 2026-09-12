"""自带 Key（BYOK）：填入 / 查看 / 轮换 / 删除自己的 LLM 配置（票 32）。

**明文 key 只进来一次，之后一步都不出去**：读接口只回尾号；`LLMConfig` 的 repr 里没有 key，
所以日志与 trace 也带不出它。删掉即立刻回落服务端全局。
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_runtime
from app.config import get_settings
from app.core.balance import VendorBalance, with_alert
from app.core.byok import LLMConfig
from app.core.cost import summarize, summarize_note
from app.core.container import Runtime
from app.core.ssrf import check_base_url, parse_allowed_hosts
from app.core.schemas import (BalanceOut, CapabilityOut, LLMConfigIn, LLMConfigOut,
                              VendorBalanceOut)
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


@router.get("/capability", response_model=CapabilityOut)
def get_capability(user: User = Depends(get_current_user),
                   rt: Runtime = Depends(get_runtime)) -> CapabilityOut:
    """看**已知**的能力结论 —— **只读缓存，不发起探测**。

    打开面板不该把模型端点探一遍，所以探测由 `POST /capability/probe` 显式触发（票 36 / #44）。
    注意「不发探测」不等于「零网络」：地址复查（票 33）会做一次 DNS 解析，与 `/llm/balance` 同。
    """
    return _capability_out(rt, user.id, do_probe=False)


@router.post("/capability/probe", response_model=CapabilityOut)
def probe_capability(user: User = Depends(get_current_user),
                     rt: Runtime = Depends(get_runtime)) -> CapabilityOut:
    """**按需**探一次（面板上的「测试连通性」）—— 强制重探，不吃缓存。探不到就如实说没探到。"""
    return _capability_out(rt, user.id, do_probe=True)


def _capability_out(rt: Runtime, user_id: str, *, do_probe: bool) -> CapabilityOut:
    cfg = rt.user_llm_config_store.get(user_id)
    if cfg is None:
        # 没配置：不是「不支持」，是「没有这回事」—— supports_tools 留 None
        return CapabilityOut(note="未配置自带模型 —— 走服务端全局模型，无需探测")
    if not rt.base_url_is_still_safe(cfg.base_url):
        # 地址复查不过（票 33）：**不去连它** —— 探测也是一条对外发请求的路径。
        # reachable 留 None：我们**没试过**，说「不通」就是编答案。
        return CapabilityOut(configured=True,
                             note="自带地址未通过安全复查，不发起探测")

    if do_probe:
        # 走 probe_fresh：点「测试连通性」必须真探一次，命中缓存也要重来
        cap, reason = rt.capability_probe.probe_fresh(cfg.base_url, cfg.api_key, cfg.model)
    else:
        cap, reason = rt.capability_probe.cached(cfg.base_url, cfg.api_key, cfg.model), ""

    if cap is None:
        if not do_probe:
            # **不写「还没探测过」**：探过但没探到结论时同样没有缓存（失败不落缓存，票 34），
            # 那句就成了假话。只说「暂无结论」—— 这句在两种情况下都成立。
            return CapabilityOut(configured=True, note="暂无结论 —— 点「测试连通性」探一次")
        # 探了但没结论：**不谎报「不支持工具」**，如实说没探到并把原因带出来
        return CapabilityOut(configured=True, checked=True,
                             note=reason or "没探到 —— 按保守默认处理（视为不支持工具）")

    # 拿到结论就说明对端**答话了**（200 或 4xx 都算），所以连通性为真
    return CapabilityOut(configured=True, checked=True, reachable=True,
                         supports_tools=cap.supports_tools, source=cap.source, note=cap.note)


@router.get("/balance", response_model=BalanceOut)
def balance(user: User = Depends(get_current_user), rt: Runtime = Depends(get_runtime)) -> BalanceOut:
    """余额与用量**两个维度分开报**，各自的来源也分开写。

    - **厂商余额**：能查则查（票 35）—— 厂商没有这个接口、或查失败，都**如实写出来**，
      绝不留空、也绝不拿下面那个「我方统计用量」冒充余额；
    - **我方统计用量**：本地记账按单价折算出来的（票 27/28），**是另一个维度**，可审计。
    """
    cfg = rt.user_llm_config_store.get(user.id)
    if cfg is None:
        vendor = VendorBalance(available=False,
                               note="未配置自带模型 —— 用的是服务端全局模型，没有厂商余额可查")
    elif not rt.base_url_is_still_safe(cfg.base_url):
        # 地址复查不过（票 33）：**不去连它** —— 余额查询也是一条对外发请求的路径
        vendor = VendorBalance(available=False, note="自带地址未通过安全复查，不发起余额查询")
    else:
        vendor = with_alert(rt.balance_probe.fetch(cfg.base_url, cfg.api_key),
                            get_settings().byok_balance_alert_threshold)
    ours = summarize(rt.usage_store.list(user.id))
    # 口径与两个维度的来源都写清楚：这是**我方统计**，不是厂商余额（票 35）
    ours["note"] = "；".join(p for p in
                             ("我方统计用量：本地记账按单价折算，不是厂商余额",
                              summarize_note(ours)) if p)
    return BalanceOut(vendor=VendorBalanceOut(**asdict(vendor)), ours=ours)
