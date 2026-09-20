# 基础镜像用**显式的 Debian 版本**，不用浮动的 `3.11-slim` —— 后者会在上游换发行版时
# 悄悄跟着变。想彻底锁死请 pin digest（本机连不上 registry，留给你在能上网的机器上跑一次）：
#     docker buildx imagetools inspect python:3.11-slim-bookworm
#   然后把下面这行换成 FROM python:3.11-slim-bookworm@sha256:<digest>
FROM python:3.11-slim-bookworm

WORKDIR /app

COPY backend/requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY backend/app ./app
COPY frontend /frontend

# **非 root 运行**：容器逃逸后拿到的不是主机 root（安全审查 F7）。
# 数据目录一并 chown 给 app —— 否则它以非 root 起来写不了 sqlite / uploaded_files；
# 首次把具名卷挂到这些目录上时，Docker 会把镜像里的属主一起带过去。
RUN useradd --create-home --shell /usr/sbin/nologin app \
 && mkdir -p /app/data /app/uploaded_files /app/logs \
 && chown -R app:app /app /frontend
USER app

EXPOSE 8000

# 健康检查用**标准库**（slim 镜像里没有 curl）。start-period 留足首次启动重建索引的时间。
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
