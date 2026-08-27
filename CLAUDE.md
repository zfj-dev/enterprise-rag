# 企业智能文档问答系统（RAG）— 项目记忆

AI 求职 + 真给身边人用的**私有化 RAG 文档问答**，单人独立开发。详见 `README.md` 与 `企业智能文档问答系统-架构文档-v1.1.md`。

## 已锁定决定（2026-08-27）
- **部署形态（W6）= A（本机自用/演示）+ B（局域网给身边人用）**
- **LLM 路线 = 混合**：本地 bge 嵌入/重排（GPU）+ 云端 API 生成
- **硬件 = RTX 4050 6G**；6G 跑不动本地大模型 → **LLM 一定走 API**

## 技术栈
- FastAPI + SQLAlchemy + pgvector（适配层，默认 InMemory）+ rank_bm25 + Redis（预留）
- 嵌入 `bge-large-zh` / 重排 `bge-reranker-large`（本地 GPU，惰性 import；默认 Fake）
- LLM：OpenAI 兼容（DeepSeek/SiliconFlow）；默认 Fake
- 解析：PyMuPDF / python-docx / openpyxl（已实现）+ Docling / PP-Structure（预留路由）
- 前端：`frontend/index.html` 单文件，FastAPI 托管，SSE 流式
- 部署：`deploy/docker-compose.yml`（演示模式：sqlite+内存向量+假模型，开箱即用）
- 测试：`pytest`（20 项：单元 + API 集成，全绿）

## 核心架构
- 分层：前端 → FastAPI（api/core/services/models/db）→ RAG 管线 → 存储
- **Ingestion**：解析路由 → Parent-Child 分块 + 上下文摘要 → 嵌入 → 向量库+BM25 入库
- **Query**：问题优化（真实模式）→ 混合检索（向量+BM25+RRF）→ 重排 → 引用校验 → LLM(SSE) → 反馈
- **权限**：检索时按 owner/kb 过滤（服务端注入），不泄漏无权限内容
- **引用**：no source → no claim；chunk 带稳定 id

## 关键实现 / 踩坑（新会话必看）
- 组件用**接口 + Fake/真实双实现**（嵌入/重排/LLM/向量库），用 `USE_REAL` / `*_PROVIDER` 切换；测试走 Fake。
- **挂载盘（E:\ai项目\*）Edit/Write 覆盖易截断/字节码错乱** → 改已有文件一律 **bash heredoc 重写**；跑测试前清 `__pycache__` 或加 `python -B`。
- **sandbox 网络慢** → 装依赖用 TUNA 镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；uvicorn 不在 PATH → 用 `python -m uvicorn`。
- **sandbox 挂载盘写 sqlite 会 disk I/O error** → 冒烟 / `DATABASE_URL` 指向 `/tmp`；真机 Windows 无此问题。
- 注册默认 `viewer`；seed 管理员 `admin/admin123`。
- 演示模式 `FakeLLM` 返回固定"（模拟回答）..."，做全链路演示；真实模式需 API Key。

## 当前进度（2026-08-27）
- ✅ 后端全链路（上传→解析→分块→向量化→混合检索→重排→生成→引用→反馈→调试）+ API 20 测试全绿 + 前端 + docker-compose + README + 本文档已完成。
- 演示模式**完整可用**；真实模式（pgvector search / Docling / 云端 LLM / 问题优化）为**适配接口就位、真机联调待做**。

## 下一步
- 真实模式：补 `PgVectorStore.search`；接入 Docling/PP-Structure；云端 LLM + 问题优化；语义缓存(Redis)；RAGAS 评估/黄金集。
- 面试讲法：混合检索+RRF+重排；引用校验；检索时权限过滤；工程取舍（pgvector/Celery/混合/砍 GraphRAG）。
