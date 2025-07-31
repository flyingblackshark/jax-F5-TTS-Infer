# F5-TTS FastAPI 推理服务

基于 [jax-F5-TTS](https://github.com/SWivid/F5-TTS) 项目的 FastAPI 推理服务，提供高性能的文本转语音 API。

## 功能特性

- 🚀 基于 JAX/Flax 的高性能推理
- 🔄 支持多种音频格式输入
- 📝 支持中英文混合文本生成
- ⚡ 异步处理和批量推理
- 🎛️ 可配置的推理参数
- 📊 健康检查和状态监控
- 🔧 RESTful API 接口

## 安装依赖

### 1. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 jax-F5-TTS 包

```bash
# 克隆并安装 jax-F5-TTS
cd /path/to/jax-F5-TTS
pip install -e .
```

### 3. 安装 JAX Vocoder

```bash
# 安装 jax_vocos
pip install git+https://github.com/SWivid/jax_vocos.git
```

## 配置

### 环境变量

可以通过环境变量配置模型路径：

```bash
export F5_MODEL_PATH="F5-TTS"  # 模型名称或路径
export VOCODER_MODEL_PATH="charactr/vocos-mel-24khz"  # Vocoder 模型路径
export VOCAB_PATH="Emilia_ZH_EN"  # 词汇表路径
export MAX_SEQUENCE_LENGTH=4096  # 最大序列长度
```

### 配置文件

编辑 `config.py` 文件来自定义模型参数：

```python
default_config = F5TTSConfig(
    pretrained_model_name_or_path="F5-TTS",
    vocoder_model_path="charactr/vocos-mel-24khz",
    vocab_name_or_path="Emilia_ZH_EN",
    max_sequence_length=4096,
    # 其他参数...
)
```

## 启动服务

### 开发模式

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 生产模式

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

**注意**: 由于 JAX 模型的内存占用，建议使用单个 worker。

## API 接口

### 健康检查

```bash
GET /health
```

响应：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 音频生成 (Form Data)

```bash
POST /generate
Content-Type: multipart/form-data
```

参数：
- `ref_text` (string): 参考音频对应的文本
- `gen_text` (string): 要生成的文本
- `ref_audio_base64` (string): 参考音频文件的 Base64 编码
- `num_inference_steps` (int, 可选): 推理步数，默认 50
- `guidance_scale` (float, 可选): 引导尺度，默认 2.0
- `speed_factor` (float, 可选): 语速因子，默认 1.0
- `use_sway_sampling` (bool, 可选): 是否使用 sway 采样，默认 false

### 音频生成 (JSON)

```bash
POST /generate_json
Content-Type: application/json
```

请求体：
```json
{
  "ref_text": "参考文本",
  "gen_text": "要生成的文本",
  "ref_audio_base64": "base64编码的音频数据",
  "num_inference_steps": 50,
  "guidance_scale": 2.0,
  "speed_factor": 1.0,
  "use_sway_sampling": false
}
```

### 响应格式

```json
{
  "audio_base64": "生成的音频数据(base64编码)",
  "sample_rate": 24000,
  "duration": 3.5,
  "generation_time": 2.1,
  "text_length": 20,
  "audio_length": 84000
}
```

### 音频下载

```bash
GET /download/{audio_id}
```

直接下载生成的音频文件。

## 使用示例

### Python 客户端

使用提供的客户端示例：

```bash
python client_example.py \
  --ref-text "这是参考文本" \
  --gen-text "这是要生成的文本" \
  --ref-audio reference.wav \
  --output generated.wav \
  --steps 50 \
  --cfg 2.0 \
  --speed 1.0
```

### cURL 示例

```bash
# 健康检查
curl -X GET http://localhost:8000/health

# 生成音频 (Form Data)
REF_AUDIO_BASE64=$(base64 -w 0 reference.wav)
curl -X POST http://localhost:8000/generate \
  -d "ref_text=这是参考文本" \
  -d "gen_text=这是要生成的文本" \
  -d "ref_audio_base64=$REF_AUDIO_BASE64" \
  -d "num_inference_steps=50" \
  -d "guidance_scale=2.0" \
  -d "speed_factor=1.0"

# 生成音频 (JSON)
curl -X POST http://localhost:8000/generate_json \
  -H "Content-Type: application/json" \
  -d '{
    "ref_text": "这是参考文本",
    "gen_text": "这是要生成的文本",
    "ref_audio_base64": "'$REF_AUDIO_BASE64'",
    "num_inference_steps": 50,
    "guidance_scale": 2.0,
    "speed_factor": 1.0
  }'
```

## 性能优化

### JAX 配置

```bash
# 设置 JAX 内存预分配
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 设置 JAX 设备
export CUDA_VISIBLE_DEVICES=0  # 使用特定 GPU
```

### 模型优化

- 使用 JIT 编译加速推理
- 支持模型分片 (sharding) 处理大模型
- 批量处理多个请求
- 缓存编译后的函数

## 故障排除

### 常见问题

1. **内存不足**
   - 减少 `max_sequence_length`
   - 使用更小的批量大小
   - 启用模型分片

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确认依赖包已正确安装
   - 查看服务器日志

3. **推理速度慢**
   - 确保使用 GPU
   - 检查 JAX 配置
   - 减少推理步数

### 日志查看

服务器会输出详细的日志信息，包括：
- 模型加载状态
- 推理时间统计
- 错误信息和堆栈跟踪

## 开发

### 项目结构

```
jax-F5-TTS-Infer/
├── main.py              # FastAPI 应用主文件
├── config.py            # 配置文件
├── client_example.py    # 客户端示例
├── requirements.txt     # 依赖列表
└── README.md           # 说明文档
```

### 扩展功能

- 添加更多音频格式支持
- 实现音频流式传输
- 添加用户认证
- 集成监控和日志系统
- 支持模型热更新

## 许可证

本项目基于原 jax-F5-TTS 项目的许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 相关链接

- [jax-F5-TTS](https://github.com/SWivid/F5-TTS)
- [F5-TTS 原项目](https://github.com/SWivid/F5-TTS)
- [JAX 文档](https://jax.readthedocs.io/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)