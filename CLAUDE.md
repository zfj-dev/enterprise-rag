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
- **UI 重构（2026-08-30）**：前端 `frontend/index.html` 重写为三栏响应式布局（左侧会话边栏 + 中央对话 + 右侧文档/参数面板），支持深色模式、会话历史列表（`GET /chat/sessions` + `/sessions/{id}`）、消息气泡 + Markdown 渲染、流式打字光标、上传进度条、快捷提问卡片、会话自动标题（取首个问题前 30 字）。聊天记录存 localStorage 刷新恢复。
- **docling 调优配置（2026-08-30）**：`docling_images_scale`(0.5 更快但小表格/图可能漏)、`docling_table_mode`(accurate/fast)、`docling_formula_enrichment`（⚠️锁定=关，勿开：Docling 本身能从文本层抽到公式原始文本并导出 `$$...$$`，企业文档公式少，不值得为公式上 CodeFormulaV2 VLM ~630MB + 解析变慢 10-40s；仅图片型公式落 `<!-- formula-not-decoded -->` 无文本）。公式分类器已支持 `$`/`\begin{}` 识别。默认保持表格质量。
- **上传提速（2026-08-30）**：docling 84s→37s（46页，do_ocr=False + GPU + converter 缓存）；bge 嵌入/重排 fp16（cuda 下）；onnxruntime-gpu 对本管线无效（docling 布局/表格模型均为 torch，已走 GPU）。
- **具体表号/图号检索修复（2026-08-30）**：
  - **① 语义缓存串台（根因）**："表3.1的内容是什么"与"表3.3的内容是什么"的 bge 向量相似度 >0.92，命中语义缓存返回**同一旧答案** → 表N 互相串。修法：`stream_answer` 对含精确编号引用（`_named_ref_intent`）的问题**跳过语义缓存**。
  - **② 强制注入 + 过滤**：具体表号（表3.2/图3.1）查询时 `_load_named_chunks` 注入含该编号的 chunk（表格优先），再**过滤只保留含该编号的候选**，LLM 看不到其他表格 → 不会选错。
  - **③ 枚举提示加强**：要求列出所有"表+数字"编号的表格（含表4.x），数量必须与列出的一一对应，不确定不报数。
  - 会话重命名/删除接口：`PATCH/DELETE /chat/sessions/{id}`。
- **黄金集评估升级（2026-08-30）**：`data/golden_set_paper.json`（10问：表3.1-3.4、枚举表格/图片、事实、页码）+ `evaluate.py`（上传论文PDF → 逐问 → 判"含期望事实/忠实度/页码"）。报告 `logs/eval-report.log`，`scripts/evaluate.ps1` 一键。**结果：答案含期望事实 100%、引用忠实度 100%、页码正确 100%**（修复前多 FAIL）。判据用 `_c()` 去空白小写，兼容 docling 的"表 4 . 1"样式。
- **公式枚举正则 bug 修复（2026-09-01）**：`_looks_like_formula` 末段正则 `\(` 的 `(` 被当成未闭合分组 → 任何问"列出公式"都报 `re.error: missing ), unterminated subpattern`。改为 `\\(`（匹配字面 `\(`）。该分支只在公式枚举时走到，早退分支（`<!-- formula-not-decoded -->`）掩盖了编译错误故单测未暴露；已补回归用例（`$$..$$`/`\begin{`/`\(`/不误判表格）。
- **公式图片→LaTeX（pix2text-mfr / TrOCR，2026-09-01）**：PDF 数学公式文本层不可靠——图片型公式无文本；文本层公式乱码（内嵌字体无 Unicode 映射，抽出变 `�`）。重活交给公式识别模型：`docling_formula_ocr=True` 时，`_parse_pdf_docling` 对 docling 标为公式但 text 为空的（not-decoded）item，用 PyMuPDF 按 bbox 裁出该区域图（docling page_no 1-based → pdf 索引 -1；bbox 转 top-left + 按页尺寸缩放），喂给公式识别模型转 LaTeX 注入 `item.text`，导出即 `$$..$$`，`_looks_like_formula` 靠 `$` 命中。模型缺装/失败一律回退占位符，不影响主流程。模型用 **`breezedeus/pix2text-mfr`**（TrOCR 架构，optimum-onnx 版）：`TrOCRProcessor` + `ORTModelForVision2Seq`，从 HF/hf-mirror 下载，兼容当前 transformers 4.5x；`optimum[onnxruntime]` 已入 requirements-real.txt。`backend/verify_formula.py` 一键验证（会同时把 `[formula]` 控制台诊断行收进报告，定位模型加载/裁图/识别哪一环）。**踩坑**：① 原 pix2tex 模型从 GitHub release 下载、国内连不上（ghproxy 也挂）→ 弃；② 换 texify(VisionEncoderDecoder) 在 transformers 4.5x 报 `'dict' object has no attribute 'to_dict'`（新 transformers 行为改动、不能降级会连带破坏 bge/docling）→ 弃；③ 最终 pix2text-mfr(TrOCR/optimum) 走 HF 镜像正常。bbox 裁图坐标以真机 verify 为准。⚠️ 需**重传文档**才会重新分块拿到 LaTeX（reindex 只从库重建索引、不重跑 docling）。
- **新账号无知识库前端修复（2026-09-02）**：新建账号 `GET /knowledge` 返回空 → 左栏知识库下拉为空、`KB=null`，`doUpload` 因 `if(!file||!KB)return` 静默失败（选文件无反应）、且有库时才能上传。三处修：① `loadKbs` 对空库自动建「默认知识库」并选中；② 知识库下拉旁加 ➕ 按钮 `createKb()`（POST /knowledge 后自动选中新建库）；③ `doUpload` 无库时toast 提示而非静默。后端建库接口 `POST /knowledge`、删除 `DELETE /knowledge/{id}` 已存在，前端此前未暴露建库按钮。
- **公式枚举：长公式截断 + 按文档序（2026-09-02）**：`$$..$$` 公式超过 child_size(128) 时被 `_split_window` 拦腰截断，`_load_typed_chunks` 加载到残缺子块（如 CE 被截在 `\log(\hat{y}_`），LLM 判"不完整"而漏排。三处修：① `chunker._split_window` 公式感知——`_formula_spans` 找 `$$..$$`/`$..$`，窗口尾若切断公式就扩到公式结束、窗口头若落公式内则跳到公式末尾，保证公式为原子块不被切；② `parser._clean_formula_latex`——压缩 pix2text 输出的连续 `\qquad` 对齐填充（232→169 字符），减 chunk 体积；③ `chat_service.prepare` 枚举时把 typed 块按 (页码,块id) 排序**前置**、公式类 enum_hint 明确"按出现顺序、残缺也当候选尽量还原、不要遗漏"。回归用例 `test_chunker.test_formula_not_cut_by_split`。⚠️ 需**重传文档**才会用新 chunker 重新分块（reindex 只重建索引、不重跑 docling/分块）。
- **前端公式渲染（KaTeX，2026-09-02）**：`renderMd` 先用正则摘出 `$$..$$`/`$..$` 数学段换占位符，`marked` 渲染 markdown 后 `DOMPurify` 清洗，再 `katex.renderToString` 回填——公式显示为排版数学而非原文。KaTeX 走 bootcdn(bootcss) 与 marked/dompurify 同源（0.16.9，HTTP 200 可达）；KaTeX 加载失败时回退显示原文。公式枚举提示词已要求 LLM 输出标准 LaTeX + 禁止 mailto 链接（`mAP@0.5` 别写成 `[mAP@0.5](mailto:...)`）。
- **前端体验（2026-09-02）**：① 左侧栏宽屏可收起/展开（`.sidebar.closed` + 汉堡常显切换，窄屏仍走 `.open`）；② 引用来源标签可点击，`showSource(i)` 弹出「文档名·第X页+引用原文」，`src-flash` 闪烁高亮定位（sources 事件已带 text）；③ 「文档库」升级为**全屏管理视图**：中间区切换为 左文档列表（库名/总数/搜索/上传/卡片[名称·页数·大小·时间·状态·选中高亮·删除，单击选中/双击预览]）+ 右预览区（PDF 用浏览器内嵌 iframe `/api/v1/documents/{id}/file`，含下载/关闭；处理中提示）；「←返回对话」恢复；后端补 `GET /documents/{id}/file`(FileResponse) + `DocumentOut` 加 `size`/`created_at`。**若点文档无反应，刷新浏览器加载新 index.html；PDF 预览需重启后端**（新接口）。
- **文档库/侧栏/预览补丁（2026-09-02）**：① PDF 预览改前端 `fetch`(带 Bearer)→objectURL 喂 iframe，解决 `/documents/{id}/file` 内嵌加载 401（iframe 无法带 Authorization 头），下载同用 blob URL；② 侧栏品牌区加「«」收起按钮，宽屏 `.sidebar.closed` 收窄、收起后靠 header ☰ 展开；③ 文档卡片加 ✏️ 重命名（后端补 `PATCH /documents/{id}` 改 filename）。
- **对话框/来源/文档库补丁（2026-09-02）**：① `#chatBox`/`.main` 加 `min-height:0` 修复 flex 下对话内容无法滚动；② 来源卡片改为**闭包绑定各自 source 对象**（历史/刷新后仍可点），`showSource(sp)` 调后端 `GET /documents/{id}/content`（按页重建文本）取**全文**、定位到引用片段 `mark.src-flash`（闪烁 2 次）+ 弹窗内 `scrollIntoView`；③ `selectKb` 选库后刷新文档库视图（docLibTitle+renderDocList）；④ 后端 `GET /documents/{id}/content` 返回按页文本。
- **前端体验（2026-09-02）**：① KaTeX 去掉 `defer`（页面脚本前加载好），消除刷新后公式延迟渲染；② `ask()` 去掉全局 `STREAMING` 锁，支持**多对话同时流式输出**，每条 AI 消息自带「⏹停止」按钮；③ 点引用来源改为**右侧「文档管理」面板切到「原文预览」模式**（`showSource`→`#srcPrev`），加载全文+自动滚动到对应页+黄色 pulse 高亮闪烁后渐隐，`‹返回` 恢复文档列表（`exitSourcePreview`）。
- **多对话并发隔离 + 原文预览精确高亮（2026-09-02）**：① `ask()` 每流独立 `AbortController`/`reqSession`/`st.session`，`ACTIVE_STREAMS` 改 Map 按流注册；`sources` 事件只在当前视图仍是该会话时才回写 `SESSION`/localStorage（防串台），并刷新会话列表；`stopCurrent()` 优先停当前激活会话(SESSION)的流、否则最后启动的，只停一个。② `showSource` 按页渲染(`src-page`)到右栏 `#srcPrev`，只高亮来源片段(`mark.src-flash`)，`prev.scrollTop+=` 手动定位到该页（仅右栏内部滚动，不滚页面）；`@keyframes srcFlash` 黄色闪烁 3 次(0/30/60%)+渐隐(2.2s ease 1)。
- **多对话 Map 架构重构（2026-09-02）**：前端会话管理层改为 `conversations: Map<uuid, Conversation>`（id/kbId/title/messages/ctrl/isStreaming/currentStreamText/sources）+ `activeConversationId`；每条消息/流式/`abortController`/`currentStreamText` 都按对话隔离；localStorage 用 `chat_history_${conversationId}` 分 key（刷新按 key 恢复、禁止合并）；新建对话即时插 UUID 标题；切换对话只改 `activeConversationId` 不关其他 SSE；停止只停当前激活对话；后端 `_get_or_create_session` 支持客户端 UUID 直接作为会话 id。左列表 pin/多选暂简化（保留重命名/删除），来源点击卡片保留。⚠️ 大重构，需真机全面回归。
