.PHONY: test test-e2e test-regression install

PY = backend/.venv/Scripts/python.exe

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
