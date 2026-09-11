"""BYOK（自带 Key）：用户配置 → LLM 实例。

本模块管两件事：
1. **按请求解析 LLM**（票 31）：工厂与配置来源都从外部注入；未配置时一律回落服务端全局，行为与今天一致。
2. **凭据的加密落库与不回显纪律**（票 32）：明文 key 只出现在「用户刚填进来」与「真要调模型」
   这两处 —— **不落库、不回显、不进日志与 trace**。

`base_url` 的私网禁入（SSRF）属于下一张票（33），不在此实现。
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache

from cryptography.fernet import Fernet

from app.core.llm import CloudLLM, LLM
from app.db.session import SessionLocal
from app.models.entities import UserLLMConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """一份自带 LLM 配置。一律按 OpenAI 兼容协议调用，不为各家新写适配器。"""

    base_url: str
    # **绝不进 repr**：日志里顺手打一个 config 就泄漏了。要看就用 `key_tail()`。
    api_key: str = field(repr=False)
    model: str


def key_tail(api_key: str) -> str:
    """只留尾号 —— 回显与排障都只用它，明文一步都不出去。"""
    k = str(api_key or "")
    return k[-4:] if len(k) >= 4 else "*" * len(k)


# 口令的最短长度：**不得有默认弱值**，也不许拿一个一字符的「口令」当加密密钥
MIN_SECRET_CHARS = 16
_SALT_BYTES = 16
_KDF_ITERATIONS = 100_000          # PBKDF2 迭代数：把「猜口令」的成本抬起来


@lru_cache(maxsize=64)
def _fernet(secret: str, salt_b64: str) -> Fernet:
    """口令 + **每行随机盐** → Fernet 密钥（PBKDF2-HMAC-SHA256），派生结果缓存。

    为什么不是裸 SHA-256：数据库文件被看到时，攻击者的下一步就是**猜口令** —— 裸摘要几乎不设成本，
    PBKDF2 让每次猜测都得真花时间；盐按行随机，同一个口令也不会得到同一把密钥。
    口令换了 = 已存的密文解不开（按未配置处理）。
    """
    key = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"),
                              base64.urlsafe_b64decode(salt_b64), _KDF_ITERATIONS)
    return Fernet(base64.urlsafe_b64encode(key))


def _seal(secret: str, plaintext: str) -> str:
    """加密并带上这一行自己的盐：`<salt>$<cipher>`。"""
    salt_b64 = base64.urlsafe_b64encode(os.urandom(_SALT_BYTES)).decode("ascii")
    return "%s$%s" % (salt_b64,
                      _fernet(secret, salt_b64).encrypt(plaintext.encode("utf-8")).decode("ascii"))


def _open(secret: str, sealed: str) -> str:
    """解封；格式不对 / 口令不对都由调用方按「未配置」处理。"""
    salt_b64, sep, cipher = str(sealed or "").partition("$")
    if not sep or not cipher:
        raise ValueError("密文格式不对")
    return _fernet(secret, salt_b64).decrypt(cipher.encode("ascii")).decode("utf-8")


class UserLLMConfigStore(ABC):
    """按用户存取自带配置。默认无配置 → 全部回落服务端全局。"""

    persistent: bool = False     # 重启后还在不在 —— 回显接口据此如实告诉用户

    @abstractmethod
    def get(self, user_id: str) -> LLMConfig | None:
        """**内部用**：拿明文 key 去构造 LLM。回显接口不许调它。"""

    @abstractmethod
    def set(self, user_id: str, cfg: LLMConfig) -> None: ...

    @abstractmethod
    def delete(self, user_id: str) -> None: ...

    @abstractmethod
    def public_view(self, user_id: str) -> dict | None:
        """**回显专用**：只给 base_url / model / key_tail / updated_at。

        做成存储层的方法，而不是「让接口自己从 get() 里把 api_key 删掉」—— 明文压根不出这一层，
        以后再加接口也不会漏。
        """


class InMemoryUserLLMConfigStore(UserLLMConfigStore):
    """内存实现：**默认**用它（谁都没配过 → 全部回落服务端全局，行为与今天一致）。

    没配 `BYOK_SECRET_KEY` 时运行时也用它 —— 凭据只活在进程里、重启即失，绝不退化成明文落库。
    """

    def __init__(self) -> None:
        self._by_user: dict[str, LLMConfig] = {}

    def get(self, user_id: str) -> LLMConfig | None:
        return self._by_user.get(user_id)

    def set(self, user_id: str, cfg: LLMConfig) -> None:
        self._by_user[user_id] = cfg

    def delete(self, user_id: str) -> None:
        self._by_user.pop(user_id, None)

    def public_view(self, user_id: str) -> dict | None:
        cfg = self._by_user.get(user_id)
        if not cfg:
            return None
        return {"base_url": cfg.base_url, "model": cfg.model,
                "key_tail": key_tail(cfg.api_key), "updated_at": ""}


class DbUserLLMConfigStore(UserLLMConfigStore):
    """加密落库实现（票 32）：库里只有**密文与尾号**，明文既不落库也不回显。"""

    persistent = True

    def __init__(self, secret: str):
        if len(str(secret or "")) < MIN_SECRET_CHARS:
            # 构造上就不给弱默认：口令为空（或短到形同虚设）一律拒绝，绝不退化成明文存储
            raise ValueError("BYOK 加密口令太短（至少 %d 字符）—— 绝不退化成明文存储"
                             % MIN_SECRET_CHARS)
        self._secret = secret

    def get(self, user_id: str) -> LLMConfig | None:
        db = SessionLocal()
        try:
            row = db.get(UserLLMConfig, user_id)
            if not row or not row.key_cipher:
                return None
            try:
                api_key = _open(self._secret, row.key_cipher)
            except Exception as e:      # noqa: BLE001 —— 口令换过 / 密文被改：按未配置处理
                logger.warning("自带 Key 解密失败（换过口令？），按未配置处理 —— 回落服务端全局：%s",
                               type(e).__name__)
                return None
            return LLMConfig(base_url=row.base_url, api_key=api_key, model=row.model)
        finally:
            db.close()

    def set(self, user_id: str, cfg: LLMConfig) -> None:
        cipher = _seal(self._secret, cfg.api_key)
        db = SessionLocal()
        try:
            row = db.get(UserLLMConfig, user_id)
            if row is None:
                row = UserLLMConfig(user_id=user_id)
                db.add(row)
            row.base_url = cfg.base_url
            row.model = cfg.model
            row.key_cipher = cipher            # **只存密文**
            row.key_tail = key_tail(cfg.api_key)
            db.commit()
        finally:
            db.close()

    def delete(self, user_id: str) -> None:
        db = SessionLocal()
        try:
            row = db.get(UserLLMConfig, user_id)
            if row is not None:
                db.delete(row)
                db.commit()
        finally:
            db.close()

    def public_view(self, user_id: str) -> dict | None:
        db = SessionLocal()
        try:
            row = db.get(UserLLMConfig, user_id)
            if row is None:
                return None
            return {"base_url": row.base_url, "model": row.model, "key_tail": row.key_tail,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else ""}
        finally:
            db.close()


class LLMFactory(ABC):
    """把一份用户配置构造成 LLM 实例（从外部注入，便于用 stub 做确定性测试）。"""

    @abstractmethod
    def build(self, cfg: LLMConfig) -> LLM: ...


class OpenAICompatLLMFactory(LLMFactory):
    """默认工厂：按 OpenAI 兼容协议构造（覆盖 DeepSeek / SiliconFlow / DashScope / OpenAI 等）。"""

    def build(self, cfg: LLMConfig) -> LLM:
        return CloudLLM(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model)
