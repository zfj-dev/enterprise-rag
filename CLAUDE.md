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

## 近期修复（2026-08-27，用户真机演示反馈）
- **中文检索增强**：FakeEmbedding 改"字符级词袋"（"营收"~"营业收入"部分匹配），不再按空格分词。
- **按页分块+真实页码**：解析器输出分页文本（`ParsedDocument.pages`），入库按页分块并标真实页码（旧上传仍是0，需重传）。
- **启动重建索引**：lifespan 调 `reindex_all`，容器重启后从数据库重建内存向量库/BM25，不用重传即可检索。
- **同名文档覆盖/跳过**：`POST /documents?overwrite=` —— 前端查当前库同名文件，确认覆盖(true)或跳过(false)；后端 overwrite 时删旧分块+索引再重传。
- **前端 KB 选择修复**：下拉框加 `onchange=selectKb()`，不再永远用第一个库；新建库后自动选中。
- 演示模式已知边界：`USE_REAL=true` 时需 LLM API Key；bge 语义检索建议在宿主跑（容器 GPU 复杂）；pgvector search 仍为适配层待做。
- **多页 PDF 卡 processing/0块（已修）：chunk id 含 page_num** —— 之前按页分块时每页都从 `_p0` 起编号，页间 id 碰撞触发 UNIQUE constraint → 事务回滚 → 卡 processing。`ParentChildChunker.chunk` id 改为 `{doc_id}_p{page_num}_{idx}`；`process_document` 异常时先 rollback 再标 failed。
- **删除文档/知识库**：`DELETE /documents/{id}`、`DELETE /knowledge/{id}`（连带删分块+检索索引），前端每文档/每库加"删"按钮；`loadDocs` 按文件名去重（同名只留最新一条）。
- 单测 22/22（新增：chunk 页间 id 唯一、删除文档/库）。
- **真实语义检索 = 宿主跑**（bge 需 GPU/torch，容器里跑 GPU 模型复杂）：`scripts/setup_real.ps1`(装 venv+torch+sentence-transformers) + `scripts/run_real.ps1`(EMBEDDING_PROVIDER=bge, RERANKER_PROVIDER=bge, LLM_PROVIDER=dashscope+Key, DEVICE=cuda)。首次启动自动下载 bge 模型并 reindex。
- config 已加 `dashscope` LLM provider + `embedding_device`/`reranker_device`(cuda, 无 GPU 自动回落 cpu)。
- **⚠️ Windows PowerShell 5.1 读 .ps1 按 ANSI/GBK，UTF-8 中文会乱码报错** → `scripts/*.ps1` 必须**纯 ASCII(英文)**（同 SW 项目 publish-sw.bat）；run_real 用 `& .venv\Scripts\python.exe -m uvicorn` 而非依赖 Activate。
- **上传异步化（2026-08-27）**：`POST /documents` 秒回 `processing`，`launch_processing` 起后台线程（`threading.Lock` 串行）解析+嵌入；前端 `pollDoc` 每 2s 轮询状态直至 indexed/failed。即使 CPU 嵌入大文档也不阻塞浏览器。测试更新为等待异步入库后断言。
- **⚠️ torch cu124 只支持 Python ≤3.12**：宿主默认 python 可能是 3.14 → 必须用 `py -3.12 -m venv .venv` 显式建 3.12 环境；`setup_real.ps1` 已强制 `py -3.12` + 阿里云 torch 镜像（download.pytorch.org 国内被墙）。
- **自检工作流（免复制粘贴反馈）**：`scripts/selftest.ps1`（ASCII，用 venv python 跑 `backend/selftest.py`）对运行中的服务 localhost:8000 跑完整链路（health/login/建库/上传/异步入库/真实LLM/引用/反馈/调试/清理），写报告到 `backend/logs/selftest-report.log`。**用户在真机跑一条命令，助手直接读该报告文件定位/修 bug，用户再跑。** 关键判据：`chat_real_llm` 若 `fake回答=True` → LLM 没接真实 provider（查 run_real 的 LLM_PROVIDER/Key/get_llm 列表）。
- ✅ **真实模式全链路跑通（2026-08-27）**：GPU bge 嵌入/重排 + DashScope qwen 真实生成 + 引用溯源；`scripts/selftest.ps1` 自检 **13/13 全绿**（health/login/建库/上传/异步入库/真实LLM/引用/反馈/调试/清理），答案示例"比亚迪2025年营业收入为803.96亿元 [来源: selftest.txt, 第1页]"。
- **P0-1 多轮对话+问题优化（2026-08-27）**：`chat_service.prepare` 先建/取会话→加载最近6轮历史→真实模式 LLM 改写(指代消解)→检索；SSE 事件带 `session_id`，前端 `SESSION` 续聊。单测 23/23。
- **P0-2 黄金集评估（2026-08-27）**：`backend/data/golden_set.json`(4问) + `backend/evaluate.py`(上传样本文档→逐问/chat→判"答案含期望事实"+"带来源"，写 `logs/eval-report.log`)。`scripts/evaluate.ps1` 一键。假模式答案事实为0属预期；真机应全过。
- **P1 引用校验升级（LLM 逐句校验，2026-08-27）**：`citation.verify_claims` 把答案按句拆分→LLM 判断每句是否被来源支撑→算引用覆盖率；真实模式生成后接入（done 事件 + trace 带 `citation_coverage`），LLM 失败降级启发式。单测 25/25。
- **P1 语义缓存（2026-08-27）**：`core/cache.py` SemanticCache（bge 向量相似 >0.92 命中，内存版默认，`redis_url`+backend=redis 走 Redis）；chat 生成前查、生成后写，done 事件/trace 带 `cache_hit`。单测 26/26。
- **P1 Docling 解析接入（2026-08-27）**：`ParserRouter` PDF 分支优先用 docling（惰性 import，表格/版面更强），失败回退 PyMuPDF；`config.parser_use_docling` 开关；`requirements-real.txt` 已含 docling。**需真机装 docling 后验证表格 PDF**。
- **Docling 真机验证通过（2026-08-29）**：论文 PDF（46页毕设《改进YOLOv8瓜果蔬菜识别》）docling 解析成功 `parser=docling` + 97 表格行 + 表3.1 命中（`verify_docling.py` 报告 PASS）。排雷四连：
  - **① additional.dat 缺失**：docling_parse 的 Windows wheel 打包丢了 `pdf_resources/glyphs/standard/additional.dat`（3.3KB 字体表）→ 读 PDF 即 `RuntimeError: filename does not exists`。**修法：`parser.py` 强制 `PdfFormatOption(backend=PyPdfiumDocumentBackend)`** 绕开 docling_parse；版面/表格仍走 StandardPdfPipeline（可考虑彻底卸 docling_parse）。
  - **② onnxruntime 没装**：docling 2.123.1 是元包、代码在 `docling-slim`，onnxruntime 属可选 extra `models-onnxruntime`；之前自检走 txt 没触发 docling 一直没暴露 → `requirements-real.txt` 补 `onnxruntime>=1.17`。
  - **③ HF 缓存 Windows 符号链接**：huggingface_hub 默认 blobs→snapshots 用 symlink，无开发者模式 → `WinError 1314` → 设 `HF_HUB_DISABLE_SYMLINKS=1`（改复制）。
  - **④ Xet 存储 401**：HF 大文件走 `cas-server.xethub.hf.co`，国内/hf-mirror 无权限 → `HF_HUB_DISABLE_XET=1` 回落普通 HTTP 走 hf-mirror。
  - ①②③④ 的 env 已写入 `run_real.ps1` / `setup_real.ps1`；`scripts/verify_docling.ps1`（py 内设置）一条命令验证任意 PDF 的 docling 解析（parser/表格行/表题 + 版本诊断 + PyMuPDF 对照），报告 `logs/docling-verify.log`。
  - ✅ **GPU torch 已装（2026-08-29 用户手动装）**：torch 2.13.0+cu130 + torchvision 0.28.0+cu130，`torch.cuda.is_available()=True`（RTX 4050）。bge 嵌入/重排真正走 GPU。`scripts/enable_gpu.ps1` 是复现脚本（阿里云 `pytorch-wheels/cu130/` 镜像，含同名 wheel）。⚠️ `setup_real.ps1` 目前**没装 `requirements-real.txt`**（docling 是手动装的），下次重建 venv 需补；且它会装 CPU torch，GPU 需另跑 enable_gpu.ps1。
- **表格检索链路打通（2026-08-29）**：论文问"表3.1的内容"返回 PyCharm/Windows11/Pytorch/CUDA12.3/RTX3060 带引用。三个修复叠加：
  - **① chunker 表格标题并入表格单元**：docling markdown 里"表 3 . 1 实验环境配置"标题和表格正文是**分开的段落**（标题+空行+表格行），导致标题和正文被切成两个 chunk、检索命不中正文。`chunker._table_aware_parents` 检测表格前的标题行（`_looks_like_caption`：去空格后以 表/图/Table/Fig+数字 开头）并**并入表格单元**，表格 chunk 同时含标识和正文。单测 `test_table_caption_merged_with_blank_line`。
  - **② BM25 中文分词**：原 `_tok` 是 `str().split()` 按空格分词，中文无空格→整句 1 个 token→BM25 对中文基本失效（"表3.1"匹配不上）。改为**去空白 + ASCII 整词 + 中文双字组**，"表3.1""比亚迪"等标识/关键词能命中。修复后表格 chunk 从纯向量 top-20 外被 BM25 拉进候选、GPU 重排顶到 #1。
  - **③ docling 提速**：`do_ocr=False`（文本 PDF 不需要 OCR，默认 True 极慢）+ `accelerator_options.device=cuda`（`DOCLING_DEVICE` 可覆盖，默认 torch.cuda.is_available 自动选）+ **模块级 converter 缓存**（`_get_docling_converter`，模型只加载一次，不再每次上传重载）。解析 46 页从 ~90s 降到 ~15-25s 预期。
- **诊断工具链（2026-08-29）**：`verify_docling.py/.ps1`（docling 解析+版本+PyMuPDF 对照）、`diagnose_table.py`（docling 输出+chunk 切分）、`diagnose_retrieval.py/.ps1`（全链路：解析→分块→索引→查询，报告 `logs/retrieval-diagnose.log`）。真机排雷与调优全走"跑脚本→读报告"模式。
- **泛化"枚举某类内容"（2026-08-29）**：问答里"给出所有表格内容/有哪些图片/列出公式"这类枚举查询，普通 top-k 检索命中不到表格/图片 chunk（它们不含"表格"等泛词）。`chat_service` 加 `_enum_intent` + `_TYPE_RULES` 泛化检测：问题含枚举信号词（所有/全部/列出/有哪些/汇总/统计…）+ 命中类别关键词（表/图/公式/代码）+ 非具体编号（表3.1/图2.1）→ 把该类别 child chunk 注入 LLM 上下文，让其枚举/汇总。加新类型只需往 `_TYPE_RULES` 加一行（关键词正则 + 具体编号排除 + chunk 判定函数）。真机验证"给出所有表格内容"枚举 5 个表。单测 `tests/test_chat_table.py`。
