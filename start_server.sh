#!/bin/bash

# F5-TTS FastAPI 服务启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认配置
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"
RELOAD="false"
LOG_LEVEL="info"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --reload)
            RELOAD="true"
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --dev)
            RELOAD="true"
            LOG_LEVEL="debug"
            shift
            ;;
        --help|-h)
            echo "F5-TTS FastAPI 服务启动脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --host HOST        绑定主机地址 (默认: 0.0.0.0)"
            echo "  --port PORT        绑定端口 (默认: 8000)"
            echo "  --workers NUM      工作进程数 (默认: 1)"
            echo "  --reload           启用自动重载 (开发模式)"
            echo "  --log-level LEVEL  日志级别 (默认: info)"
            echo "  --dev              开发模式 (等同于 --reload --log-level debug)"
            echo "  --help, -h         显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  F5_MODEL_PATH      F5 模型路径"
            echo "  VOCODER_MODEL_PATH Vocoder 模型路径"
            echo "  VOCAB_PATH         词汇表路径"
            echo "  MAX_SEQUENCE_LENGTH 最大序列长度"
            echo ""
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查 Python 环境
print_info "检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 未找到，请先安装 Python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python 版本: $PYTHON_VERSION"

# 检查依赖
print_info "检查依赖包..."
required_packages=("fastapi" "uvicorn" "jax" "flax")
for package in "${required_packages[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        print_error "缺少依赖包: $package"
        print_info "请运行: pip install -r requirements.txt"
        exit 1
    fi
done
print_success "所有依赖包已安装"

# 检查 JAX 设备
print_info "检查 JAX 设备..."
JAX_DEVICES=$(python3 -c "import jax; print(f'设备数量: {jax.device_count()}, 设备类型: {jax.devices()[0].device_kind if jax.devices() else "无"}'" 2>/dev/null || echo "JAX 设备检查失败")
print_info "$JAX_DEVICES"

# 设置环境变量
print_info "设置环境变量..."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    print_info "使用 GPU: $CUDA_VISIBLE_DEVICES"
else
    print_warning "未设置 CUDA_VISIBLE_DEVICES，将使用所有可用 GPU"
fi

# 显示配置信息
print_info "服务配置:"
echo "  主机: $HOST"
echo "  端口: $PORT"
echo "  工作进程: $WORKERS"
echo "  自动重载: $RELOAD"
echo "  日志级别: $LOG_LEVEL"
echo ""

# 显示模型配置
print_info "模型配置:"
echo "  F5 模型: ${F5_MODEL_PATH:-F5-TTS}"
echo "  Vocoder 模型: ${VOCODER_MODEL_PATH:-charactr/vocos-mel-24khz}"
echo "  词汇表: ${VOCAB_PATH:-Emilia_ZH_EN}"
echo "  最大序列长度: ${MAX_SEQUENCE_LENGTH:-4096}"
echo ""

# 构建 uvicorn 命令
UVICORN_CMD="uvicorn main:app --host $HOST --port $PORT --log-level $LOG_LEVEL"

if [[ "$RELOAD" == "true" ]]; then
    UVICORN_CMD="$UVICORN_CMD --reload"
else
    UVICORN_CMD="$UVICORN_CMD --workers $WORKERS"
fi

# 启动服务
print_success "启动 F5-TTS FastAPI 服务..."
print_info "命令: $UVICORN_CMD"
print_info "服务地址: http://$HOST:$PORT"
print_info "API 文档: http://$HOST:$PORT/docs"
print_info "健康检查: http://$HOST:$PORT/health"
echo ""
print_info "按 Ctrl+C 停止服务"
echo ""

# 执行启动命令
exec $UVICORN_CMD