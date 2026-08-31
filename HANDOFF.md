# 会话交接 — 2026-08-29

## 刚完成
- **Docling 论文表格检索全链路打通**：46页毕设论文《改进YOLOv8瓜果蔬菜识别》问"表3.1的内容是什么" → 返回 PyCharm/Windows11/Pytorch/CUDA12.3/RTX3060 带来源引用。
- **GPU 生效**：torch 2.13.0+cu130（用户手动装），bge 嵌入/重排 + docling 版面模型走 GPU。
- **真实模式自检 13/13、黄金集 4/4 全绿**（含 GPU 后复跑）。

## 排雷记录（详见 CLAUDE.md）
- docling_parse `additional.dat` 缺失 → `parser.py` 强制 PyPdfium 后端
- `onnxruntime` 缺失 → requirements-real.txt 补 `onnxruntime>=1.17`
- HF 缓存 Windows 符号链接 `WinError 1314` → `HF_HUB_DISABLE_SYMLINKS=1`
- Xet 存储 401 → `HF_HUB_DISABLE_XET=1`
- 表格标题与正文分离 → chunker 标题并入表格单元（单测 `test_table_caption_merged_with_blank_line`）
- BM25 中文分词失效 → 去空白 + ASCII 整词 + 中文双字组
- docling 慢 → `do_ocr=False` + `DOCLING_DEVICE=cuda` + 模块级 converter 缓存

## 新增脚本/工具（真机排雷/调优全走"跑脚本→读报告"）
- `scripts/`: verify_docling.ps1, diagnose_table.ps1, diagnose_retrieval.ps1, enable_gpu.ps1, fix_docling.ps1, selftest.ps1, evaluate.ps1
- `backend/`: verify_docling.py, diagnose_table.py, diagnose_retrieval.py, selftest.py, evaluate.py
- 报告落在 `backend/logs/*.log`
- 测试 28/28 全绿

## 下一步
- 重启 run_real → 重新上传论文体验 docling 提速（~90s → 15-25s 预期）
- 路线图 P1 余项：pgvector 持久化 或 前端 Markdown 渲染升级
- 泛化枚举（所有表格/图片/公式…）已支持（chat_service._enum_intent + _TYPE_RULES）
- 已知遗留：docling 解析页码全为"第1页"（markdown 导出不保留分页），引用页码不准，后续优化

## 注意
- **run_real.ps1 的 DashScope Key 已打码提交**，真机需重新粘贴自己的 Key 才能跑真实模式
- `backend/paper.pdf` 与 `backend/data/golden_set.json` 均被 gitignore，未提交
