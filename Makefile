.PHONY: test test-e2e test-regression install hooks

# 路径是**相对 backend/** 的 —— 每条 recipe 都先 cd backend。
# （原先写成 backend/.venv/...，cd 之后就成了 backend/backend/.venv/...，一个目标都跑不起来。）
# 这套是 Windows 的 venv 布局；Linux/macOS 上换成 .venv/bin/python。
PY = .venv/Scripts/python.exe

# 全部后端测试（单元 + API + 回归契约）
test:
	cd backend && $(PY) -B -m pytest tests/ -q

# 回归契约（已知缺陷用 xfail 固化；修复后应移除标记）
test-regression:
	cd backend && $(PY) -B -m pytest tests/test_regression.py -v

# 端到端 Playwright（需先起服务 localhost:8000）
test-e2e:
	cd backend && $(PY) -B ../e2e/test_e2e_smoke.py

install:
	cd backend && $(PY) -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启用提交前钩子（提交前跑全量测试）。
# **新克隆必须先跑一次** —— `core.hooksPath` 是本地 git 配置、不随仓库分发，
# 不设的话 `.githooks/pre-commit` 形同不存在，谁都不会发现自己没被保护。
hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit 钩子已启用（提交前会跑全量测试）"
