# 企业智能文档问答系统（RAG）

> 私有化部署的**检索增强生成（RAG）文档问答**：上传文档 → 解析 / 分块 / 向量化 → 自然语言问答 → **逐句引用溯源**。
> 单人独立开发，目标岗位 **AI 应用 / Agent 工程**；部署形态面向「本机自用 + 局域网给身边人用」。

上传一份 PDF，然后问「表 3.1 的实验环境是什么」「列出所有表格」「营业收入是多少」——回答带**可点击的原文出处（文档名 + 页码）**，点击即跳转并在原文中高亮定位。

**技术路线：混合。** 嵌入与重排跑本机 / 私有 GPU 上的 `bge` 小模型，生成走云端 LLM API —— 因为 4050 的 6G 显存放得下嵌入/重排，放不下一个能用的本地大模型（详见文末「设计取舍」）。

---

## 快速开始（演示模式，零依赖）

```bash
cd deploy
docker compose up -d
# 打开 http://localhost:8000  →  默认账号 admin / admin123
```

演示模式全离线（`sqlite + 内存向量 + Fake 嵌入/重排/LLM`），开箱即可走通完整链路：
**上传 → 解析 → 分块 → 向量化 → 混合检索 → 重排 → 生成 → 引用 → 反馈 → 调试**。

<details>
<summary>本地开发（不用 Docker）</summary>

```bash
cd backend
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

</details>

---

## 三种运行形态

| | ① 演示模式（默认） | ② 本机真实模式 | ③ 云端编排模式 |
|---|---|---|---|
| 场景 | 任何人 clone 即跑 | 有本地 GPU 的笔记本 | 无 GPU 的 CPU VPS |
| 嵌入 / 重排 | Fake（字符词袋） | `bge` 本机 GPU（fp16） | `api` → 私有推理节点 |
| LLM 生成 | Fake | 云端 API | 云端 API |
| 存储 | sqlite + 内存向量 | sqlite + 内存向量（可切 pgvector） | Postgres(+pgvector) + Redis |
| 启动 | `docker compose up -d` | `scripts\run_real.ps1` | `scripts\run_cloud.ps1` |
| 需要 | 无 | GPU + `requirements-real.txt` + API Key | 私有 GPU 节点 + API Key |

形态之间只切环境变量（见 [.env.example](.env.example)）：
`USE_REAL` / `EMBEDDING_PROVIDER` / `RERANKER_PROVIDER` / `LLM_PROVIDER` / `VECTOR_STORE`。

Provider 取值：`EMBEDDING_PROVIDER` / `RERANKER_PROVIDER` = `fake` | `bge`（本地 GPU）| `api`（自建推理节点）| `siliconflow`（托管，**无 GPU 也能跑真实检索**）；`LLM_PROVIDER` = `fake` | `deepseek` | `siliconflow` | `dashscope` | `openai`；`VECTOR_STORE` = `inmemory` | `pgvector`。

> **换嵌入模型 = 换口径**：托管那档默认配 `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3`（1024 维，与 `EMBEDDING_DIM` 默认值一致）。
> 换了模型之后，**历史报告的检索数字不能直接比**（报告里的配置快照会记下用的哪个），且**已入库的文档向量空间变了、需要重传**。
> 嵌入返回空向量或维度与 `EMBEDDING_DIM` 不符时**直接报错**，不会静默降级 —— 那会让内容悄悄检索不到。

---

## 架构

```
前端 (frontend/index.html 单文件, FastAPI 托管, SSE 流式)
   |
FastAPI (无状态)  -- 鉴权 / 知识库 / 文档 / 问答 / 反馈 / 调试 / 指标
   |
   +- 写入链路  ParserRouter -> Parent-Child 分块(+上下文摘要) -> 嵌入 -> 向量库 + BM25
   |                +- PDF: Docling(版面/表格) -> 回退 PyMuPDF;  公式图 -> pix2text 转 LaTeX
   |
   +- 查询链路  问题优化(多轮指代消解) -> 混合检索(向量+BM25+RRF) -> 重排
                -> 引用校验 -> LLM 生成(SSE) -> 反馈
   |
本机/私有 GPU: bge-large-zh-v1.5 嵌入 + bge-reranker-large 重排   (也可走私有推理节点 API)
云端 API:      LLM 生成 (DashScope / DeepSeek / SiliconFlow)
存储:          PostgreSQL + pgvector (关系+向量) / sqlite+内存(默认) + Redis(可选) + 文件系统
```

完整设计（数据模型、管线细节、演进路径）见 [企业智能文档问答系统-架构文档-v1.1.md](企业智能文档问答系统-架构文档-v1.1.md)。

---

## 核心能力

**解析**

- 解析路由 `ParserRouter`：PPTX / DOCX / XLSX / 文本走原生解析；PDF 优先 **Docling**（版面 + 表格结构），失败回退 PyMuPDF。
- **公式图片 → LaTeX**：对 Docling 标为公式、但文本层为空的 item，用 PyMuPDF 按 bbox 裁图 → `pix2text-mfr`（TrOCR / optimum-onnx）识别 → 注入 `$$...$$`；模型缺失或识别失败一律回退占位符，不阻塞主流程。
- 输出**分页文本**，入库保留**真实页码**（引用页码可信）。

**分块**

- Parent-Child：**child** 供检索（128，更聚焦）、**parent** 供生成（512，上下文更全），并前置一段上下文摘要。
- **表格标题并入表格单元**：`表 3.1 实验环境配置` 与表格正文常被切成两块导致检索命不中，故检测标题行并合入同一块。
- **公式原子块**：`$$...$$` 不被切窗截断。

**检索**

- **混合检索**：向量（bge）+ **BM25**（中文按**双字组**分词，解决中文无空格导致 BM25 近乎失效）+ **RRF** 融合 → **重排**。
- **权限在检索时下推**：owner / kb 过滤由服务端注入向量检索表达式，而非生成后再判断 —— 不泄漏无权限内容。
- **枚举意图**：`列出所有表格 / 有哪些图片 / 列出公式` 这类泛化枚举，按类型注入对应 chunk 供 LLM 汇总（普通 top-k 命中不到它们）。
- **精确编号**：`表 3.1 的内容` 强制注入含该编号的 chunk 并**过滤其余候选**，避免 LLM 选错表。
- **问题优化**：多轮会话下用 LLM 做指代消解 / 查询改写（`它营收呢？` → `比亚迪的营收？`）。

**生成与引用**

- SSE 流式输出，逐 token 打字效果。
- **no source → no claim**：无来源不得断言；生成后**逐句 LLM 引用校验**（该论断是否真被来源支撑），聚合出**引用覆盖率**。
- 引用可点击 → 右栏**原文预览**：按页渲染全文、滚动定位到引用页、命中片段高亮闪烁。

**工程**

- **异步上传**：`POST /documents` 秒回 `processing`，后台线程串行解析 + 嵌入（大文档不阻塞浏览器），前端轮询状态至 `indexed` / `failed`。
- **语义缓存**：bge 相似度 > 0.92 命中（内存版默认，`REDIS_URL` 可切 Redis）；**含精确编号的问题跳过缓存**，否则「表 3.1」与「表 3.3」向量高度相似会互相串台。
- **鉴权与限流**：JWT + RBAC（`admin` / `uploader` / `viewer`）、登录限流、**SSE 每用户并发上限**。
- **可观测**：per-query 全链路 trace + 调试面板（`/api/v1/debug/query`）；全局异常与前端 JS 错误落盘 `logs/error.log`。
- **自带 Key（BYOK）**：左栏「🔑 自带模型」里填自己的 OpenAI 兼容端点 / Key / 模型名，问答即按**发起用户**解析用谁的模型 —— 花他自己的钱，且**不占服务端额度**（豁免前提是「配置存在**且**地址复查通过」，否则会白送额度）。
  - Key **加密落库、只回尾号**（PBKDF2-HMAC-SHA256 派生 + Fernet，逐行随机盐）；明文不落库、不回显、不进日志；没配 `BYOK_SECRET_KEY` 时**只存内存**并如实告知「重启即失」。
  - `base_url` 过 **SSRF 防护**（默认只收 `https`、禁私网 / 回环 / 链路本地），保存与使用**各验一道**。
  - 面板同时显示**厂商余额**与**我方统计用量**两个维度 —— 厂商没有这个接口或查失败时，**照实写原因**，不拿用量冒充余额。
  - 换 / 删即时生效，删掉即回落服务端全局模型。

**前端**（[frontend/index.html](frontend/index.html)，单文件、无构建）

- 三栏响应式 + 深色模式；**多会话并行流式**（按会话隔离，互不串台），每条回答可单独停止。
- **KaTeX** 渲染公式；引用点击定位；文档库管理（重命名 / 删除 / PDF 内嵌预览）。

---

## 评测

三套脚本，全部「跑一条命令 → 读报告文件」，不需要人工比对：

| 脚本 | 测什么 | 报告 |
|---|---|---|
| [backend/evaluate.py](backend/evaluate.py) | **端到端**：答案含期望事实 / 引用忠实度（事实确在来源里）/ 引用页码正确 | `backend/logs/eval-report.log` |
| [backend/evaluate_retrieval.py](backend/evaluate_retrieval.py) | **离线检索**：混合 / 纯向量 / 纯 BM25 的 hit@k、recall@k、MRR + 负样本拒答 | `backend/logs/retrieval-eval-report.log` |
| [backend/evaluate_latency.py](backend/evaluate_latency.py) | **并发延迟**：N 并发同时打 `/chat/stream`，检索 / 重排 / 生成分三段 + TTFT（含检索与重排） | `backend/logs/latency-report.log` |
| [backend/evaluate_rgb.py](backend/evaluate_rgb.py) | **RGB 中文四能力**：噪声鲁棒 / 否定拒绝 / 信息集成 / 反事实鲁棒 | `backend/logs/rgb-eval-report.log` |
| [backend/evaluate_agent.py](backend/evaluate_agent.py) | **代理链路**：开了代理开关后事实命中 / 延迟的**变化** | `backend/logs/agent-eval-report.log` |
| [backend/evaluate_guardrail.py](backend/evaluate_guardrail.py) | **压缩质量护栏**：压缩前后同跑，事实命中不下降 | `backend/logs/guardrail-report.log` |
| [backend/evaluate_memory.py](backend/evaluate_memory.py) | **跨会话召回**：记忆注入的示例与覆盖率 | `backend/logs/memory-eval-report.log` |
| [backend/selftest.py](backend/selftest.py) | **全链路自检**：health / 登录 / 建库 / 上传 / 异步入库 / 真实 LLM / 引用 / 反馈 / 调试 / 清理 | `backend/logs/selftest-report.log` |

[backend/evaluate_all.py](backend/evaluate_all.py)（`scripts\evaluate_all.ps1`）把它们合成**一页报告** → `backend/logs/eval-summary.log`，开头附配置快照与目标线。

黄金集在 [backend/data/](backend/data/)（问题 + 期望事实 + 页码 + 负样本）。运行：`scripts\evaluate.ps1`（需先起服务）。

**最近一次真机结果**（46 页论文 PDF，GPU bge + DashScope `qwen-plus`，10 问）：

| 指标 | 结果 |
|---|---|
| 答案含期望事实 | **100%** (10/10) |
| 引用忠实度（事实在来源中） | **100%** (10/10) |
| 引用页码正确 | **100%** (10/10) |
| 全链路自检 | **13/13** |

> **口径**：判据为「去空白、小写后子串匹配」，兼容 Docling 在数字与标点间插入空格（如 `表 4 . 1`）。
> `evaluate.py` 默认取 `backend/paper.pdf`（自备文档，未入库）；跑自己的文档请设 `EVAL_DOC`。

> **一页报告的现状**：上表是**端到端那一节**的真机数字（46 页论文 PDF，GPU bge + DashScope `qwen-plus`）。
> 生成层 / 延迟 / RGB 三节在一页报告里目前写的是**「未跑」** —— 缺的是运行前置（服务要在跑、
> RGB 要官方 `data/`、并发要调高 `MAX_CONCURRENT_STREAMS_PER_USER`），**缺前置不产假数字**。

---

## 测试

```bash
cd backend
python -m pytest tests/ -q        # 601 passed, 3 skipped（单元 + API 集成 + 回归契约，约 150s）
```

3 项 skip 对应明确未实现的功能：refresh token、登出黑名单、pgvector（需 `TEST_PG_URL` 指向真实库）。

前端 E2E（Playwright，需先起服务）：

```bash
cd frontend
npm install && npx playwright install
npm run e2e
```

也可用 `make test` / `make test-regression` / `make test-e2e`。

---

## 部署

**① 演示 / 局域网给身边人用** — [deploy/docker-compose.yml](deploy/docker-compose.yml)

```bash
cd deploy && docker compose up -d
```

- 后端绑 `0.0.0.0`，同 WiFi 下访问 `http://<内网IP>:8000`（`ipconfig` 查内网 IP）。
- ⚠️ **Windows 需放行入站端口 8000** —— 局域网访问最常卡在这一步（防火墙 → 高级设置 → 入站规则 → 新建端口规则）。
- 各自注册账号；知识库按 `owner` 隔离，检索时过滤。

**② 生产编排** — [deploy/docker-compose.prod.yml](deploy/docker-compose.prod.yml) + [deploy/Caddyfile](deploy/Caddyfile)

`PostgreSQL(pgvector) + Redis + API + Caddy(HTTPS 反代，SSE 不缓冲)`；密钥走同目录 `.env`（模板 [deploy/.env.example](deploy/.env.example)）。一键：

```bash
./scripts/deploy.sh       # git pull -> build -> up -d -> 等 health；备份见 scripts/backup.sh
```

> 当前生产编排用 `VECTOR_STORE=inmemory`（单 worker，启动 `reindex_all` 从库重建索引）。
> `PgVectorStore.search` 已实现并有单测覆盖，切换前建议真机验证。
> 安全：`USE_REAL=true` 时**强制**要求非默认 `SECRET_KEY`（启动即校验），CORS 默认收紧为显式白名单。

**③ 私有推理节点** — [inference_service/app.py](inference_service/app.py)

把 `bge` 嵌入 + 重排搬到**私有云 GPU**，编排机不再需要任何 GPU：

- 提供 `POST /embed`、`POST /rerank`；`X-Inference-Token` 共享密钥鉴权。
- 模型惰性加载 + fp16 + 结果缓存 + 并发信号量；只监听私网。
- 启动：`scripts\run_inference_node.ps1`；编排侧用形态 ③ 的 `EMBEDDING_PROVIDER=api` 指向它。

它的实际用途是把嵌入/重排从本机迁到另一台 GPU 机器（或云 GPU），编排机本身不必带 GPU。**注意：生成仍走云端 LLM API**，所以这**不构成「数据不出内网」的保证**。

---

## 目录结构

```
enterprise-rag/
├── backend/
│   ├── app/
│   │   ├── api/v1/        # auth / knowledge / documents / chat / feedback / debug / metrics
│   │   ├── core/          # parser / chunker / embedding / retriever / reranker / llm
│   │   │                  #   / citation / cache / prompt / vector_store / bm25 / container
│   │   ├── models/        # SQLAlchemy ORM 实体
│   │   ├── services/      # document_service / chat_service
│   │   └── db/  utils/  config.py  main.py
│   ├── data/              # 黄金集（样本文档由 scripts/gen_sample_docs.py 生成）
│   ├── tests/             # pytest（单元 + API 集成 + 回归契约）
│   ├── evaluate.py  evaluate_retrieval.py  selftest.py  verify_*.py  diagnose_*.py
│   └── requirements.txt  requirements-real.txt  requirements-prod.txt  Dockerfile
├── frontend/              # index.html 单文件 UI + e2e/ (Playwright)
├── deploy/                # docker-compose.yml / .prod.yml / Caddyfile
├── inference_service/     # 私有 GPU 推理节点
├── scripts/               # PowerShell 一键脚本（setup_real / run_real / run_cloud / evaluate ...）
├── e2e/                   # 接口扫描与端到端冒烟
├── docs/                  # ROADMAP 等
└── CONTEXT.md  CLAUDE.md  # 领域词汇表 / 项目记忆
```

---

## 设计取舍（面试可讲）

- **为什么混合（本地嵌入 + 云端 LLM）？** 4050 的 6G 显存放得下 `bge` 嵌入/重排，放不下能用的本地 LLM。这是**按硬件约束做的取舍**，不是偷懒：留 `*_PROVIDER` 路由，换本地模型或别家 API 都只改环境变量。
- **为什么 pgvector 而不是 Milvus？** 单机、<1000 万向量、且已在跑 Postgres 时，pgvector 零新增基建、一套备份。保留 `VectorStore` 适配层，规模化后再按接口切 Milvus / Qdrant —— **做对取舍比堆基建更体现工程能力**。
- **为什么核心链路不用 LangChain？** 高频路径（检索 / 重排 / 生成）用原生 SDK，抽象透明、生产可排障；只在需要跨多工具编排时才参考图式编排思想。
- **为什么异步不用 Celery？** 单机文档量下后台线程足够，省掉 broker / worker / DLQ 三件套；留了换队列的抽象。
- **砍掉过度设计**：GraphRAG / 多模态 / Agentic / 多租户 —— 等到**有失败模式数据支撑**才往上爬（见 [docs/ROADMAP.md](docs/ROADMAP.md)）。

---

## 已知边界与路线图

- ✅ **已完成**：全链路（上传→检索→引用→反馈）、混合检索 + RRF + 重排、逐句引用校验、评测闭环、Agentic（ReAct + 3 工具 + 引用自检）、MCP server（可被任意 MCP 客户端挂载）、跨会话记忆、**上下文压缩**（滚动摘要 + 工具结果清理 + 压缩质量护栏）、**成本可见性**（per-query 用量/费用 + 额度硬拦）、**自带 Key（BYOK）**（加密落库 + SSRF 防护 + 能力探测 + 余额提醒 + 前端配置面板）、私有推理节点、生产编排。
- 🚧 **待补的交付物**（详见 [docs/ROADMAP.md](docs/ROADMAP.md)）：**带数字的一页评测报告**（代码已全部就位，缺的是运行前置，见上）、**在线 Demo 地址与演示视频**。
- ⚠️ **边界**：
  - **在线 Demo 地址与演示视频尚未上线**（本仓库目前是源码 + 一键本地/局域网部署）。
  - 真实模式依赖 `requirements-real.txt`（`bge` / `docling` / 公式 OCR 模型）；Windows 上 Docling 下载 HuggingFace 模型需要 `run_real.ps1` 里预设的几个环境变量（关闭符号链接、关闭 Xet、走镜像）。
  - 生产编排的 pgvector 切换尚待真机验证。

### 挂载 MCP（可选：让 Claude Desktop / Claude Code 直接用这些工具）

三个工具（`KbRetrieve` / `SqlQuery` / `Calculator`）实现只有一份：进程内给代理调，也能挂成 MCP
server 给任意客户端调。默认走本地 stdio，不占端口、不出网络。

```json
"enterprise-rag": {
  "command": "<backend>/.venv/Scripts/python.exe",
  "args": ["-m", "app.mcp.server", "--user", "admin", "--kb", "<kb_id>"],
  "cwd": "<backend>",
  "env": {"DATABASE_URL": "sqlite:///./rag.db"}
}
```

`--user` / `--kb` 决定这次挂载的**身份与范围**（服务端配置注入，客户端传什么都不看）；`env` 要给全
—— MCP SDK 默认只把一份白名单环境交给子进程，少了 `DATABASE_URL` 挂载点会当场退出。挂上之前先跑
`scripts/verify_mcp.ps1` 做一次真机往返自检（报告 `backend/logs/mcp-verify.log`）。

对外暴露（HTTP）默认关闭：要同时设 `MCP_TRANSPORT=http` + `MCP_ALLOW_NETWORK=true` +
`MCP_TOKEN=<随机串>` 才启动，且不写 `MCP_HOST` 就只绑 127.0.0.1。


---

## 文档索引

| 文件 | 内容 |
|---|---|
| [CONTEXT.md](CONTEXT.md) | **领域词汇表** —— 术语以此为准 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 要加什么 / 顺序 / 明确不做 |
| [企业智能文档问答系统-架构文档-v1.1.md](企业智能文档问答系统-架构文档-v1.1.md) | 完整设计（数据模型 / 管线 / 取舍 / 演进） |
| [CLAUDE.md](CLAUDE.md) | 项目记忆与踩坑记录（新会话先读） |
