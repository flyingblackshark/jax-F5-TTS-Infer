FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装 jax-F5-TTS (需要根据实际情况调整)
# RUN pip install git+https://github.com/SWivid/F5-TTS.git

# 安装 jax_vocos
RUN pip install git+https://github.com/SWivid/jax_vocos.git

# 复制应用代码
COPY . .

# 设置环境变量
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV JAX_ENABLE_X64=true
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]