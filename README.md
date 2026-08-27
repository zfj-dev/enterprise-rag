# 企业智能文档问答系统（RAG）

私有化、可单人落地、可局域网给身边人用的**检索增强生成（RAG）文档问答系统**。上传文档 → 自动解析/分块/向量化 → 自然语言问答 → **带引用溯源**。

> 面向「本机自用 + 局域网给身边人用（A+B）」的单机部署，实测运行在 **RTX 4050（6G）** 笔记本，LLM 走**混合**：本地 bge 嵌入/重排 + 云端 API 生成。

---

## 快速开始（一键）

```bash
cd deploy
docker-compose up -d
# 浏览器打开 http://localhost:8000  → 默认账号 admin / admin123
```

> 演示模式（默认）全离线可用（sqlite + 内存向量 + 假模型），就能看完整的 上传→检索→问答→引用→反馈 闭环。

**本机直接跑（开发）**：

```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## 架构概览

```
前端(index.html, FastAPI 托管, SSE)
   │
FastAPI (无状态) ─ 上传/问答/知识库/反馈/调试/指标
   │
RAG 管线: 问题优化 → 混合检索(向量+BM25+RRF) → 重排 → 引用校验 → LLM 生成
   │
本机 GPU: bge-large-zh 嵌入 + bge-reranker-large 重排   (混合模式)
本机 CPU: Docling/PP-Structure 解析(异步)
云端 API: LLM 生成 (DeepSeek / SiliconFlow)
   │
存储: PostgreSQL + pgvector (关系+向量+BM25) + Redis + 文件系统
```

完整设计（含数据模型、管线、取舍、演进路径）见 `../企业智能文档问答系统-架构文档-v1.1.md`（项目根）。

---

## 两种运行模式

| | 演示模式（默认） | 真实模式 |
|---|---|---|
| 数据库 | sqlite + 内存向量 | PostgreSQL + pgvector |
| 嵌入/重排 | Fake（词袋） | bge-large-zh / bge-reranker-large（本地 GPU） |
| LLM | Fake | DeepSeek / SiliconFlow 云端 API |
| 需要 | 无 | GPU + API Key + 装 `requirements-real.txt` |

**演示模式**开箱即用，用于看完整链路；**真实模式**切换 `USE_REAL=true` 并把 `.env.example` 里的 `EMBEDDING_PROVIDER=bge`、`RERANKER_PROVIDER=bge`、`LLM_PROVIDER=deepseek`、`LLM_API_KEY=...`、`VECTOR_STORE=pgvector`，再：

```bash
pip install -r requirements-real.txt      # GPU 依赖 + pgvector 驱动
```

> ⚠️ 真实模式当前边界：`pgvector` 向量检索与 Docling/PP-Structure 解析为**适配接口已就位、真机联调待做**（见架构文档 §15 演进路径）。演示模式已验证完整链路。

---

## 局域网给身边人用（A+B）

1. 后端绑 `0.0.0.0`（docker-compose 已默认绑定；本机跑用 `--host 0.0.0.0`）。
2. 查本机内网 IP：`ipconfig`（Windows）→ 如 `192.168.1.20`。
3. **Windows 防火墙放行端口**（入站）——这是局域网访问最常卡住的点：
   - 控制面板 → Windows Defender 防火墙 → 高级设置 → 入站规则 → 新建规则 → 端口 `8000` → 允许。
4. 身边人同 WiFi 访问 `http://你的内网IP:8000`。
5. 各自注册账号；每个知识库归属其创建者，检索时按 `owner` 过滤（权限在检索层下推，不泄漏无权限内容）。

---

## 目录结构

```
enterprise-rag/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # 路由(auth/knowledge/documents/chat/feedback/debug/metrics)
│   │   ├── core/            # parser/chunker/embedding/retriever/reranker/llm/citation/prompt/vector_store
│   │   ├── models/          # SQLAlchemy 实体
│   │   ├── services/        # document_service / chat_service
│   │   ├── db/  utils/  config.py  main.py
│   ├── tests/               # pytest（单元 + API 集成）
│   ├── requirements.txt requirements-real.txt Dockerfile
├── frontend/index.html      # 自托管 Web UI（无构建）
├── deploy/docker-compose.yml
├── docs/
└── CLAUDE.md                # 项目记忆（自动加载）
```

---

## 关键设计（面试可讲）

- **混合检索**：向量(pgvector) + BM25 + RRF 融合 + bge 重排（重排是检索质量最高 ROI 的一步）。
- **引用严谨**：`no source → no claim`；检索结果带稳定 chunk_id；答案引用可追溯原文片段。
- **权限在检索时过滤**（服务端注入 owner/kb，向量检索表达式下推），不是生成后再判断。
- **用户反馈闭环**：点踩落库 → 进黄金测试集 → 反哺评估。
- **可观测**：per-query 全链路 trace + 调试面板（`/api/v1/debug/query`）。
- **工程取舍**：pgvector 不硬堆 Milvus；APScheduler 不用 Celery；本地小模型 + 云端 API 混合（6G 跑不动本地大模型）。
- **砍掉过度设计**：GraphRAG / 多模态 / Agentic / 多租户 SaaS（有明确失败模式才往上爬）。

详见 `../企业智能文档问答系统-架构文档-v1.1.md`。

---

## 测试

```bash
cd backend
python -m pytest tests/ -v                # 单元 + API 集成
```

---

## 下一步（真实模式）

1. pgvector 向量检索落地（`core/vector_store.py` 的 `PgVectorStore` 补 `search`）。
2. Docling/PP-Structure 接入解析路由（`core/parser.py`）。
3. 问题优化（LLM 改写）在真实模式启用（`chat_service._maybe_rewrite` 已留）。
4. 语义缓存（Redis）+ 评估管线（RAGAS / 黄金集）。
