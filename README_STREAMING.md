# F5-TTS 流式传输服务

本项目基于 JetStream 架构实现了 F5-TTS 的流式传输功能，支持实时音频生成和推理引擎模式。

## 架构概述

### 核心组件

1. **TTSEngine**: 基于 JetStream 的推理引擎，支持在线和离线模式
2. **流式传输**: 支持实时音频块传输，减少延迟
3. **异步处理**: 使用多线程处理请求队列、预处理、推理和后处理
4. **生命周期管理**: 使用 FastAPI lifespan 管理引擎初始化和清理

### 工作模式

- **在线模式 (Online)**: 支持流式传输，实时返回音频块
- **离线模式 (Offline)**: 批量处理，返回完整音频

## API 端点

### 1. 健康检查
```
GET /v1/health
```

返回服务状态，包括 F5TTS 编排器和推理引擎的状态。

### 2. 流式生成
```
POST /v1/generate/stream
```

支持流式传输的音频生成端点。返回 Server-Sent Events (SSE) 格式的音频块。

**请求体**:
```json
{
  "ref_text": "参考文本",
  "gen_text": "要生成的文本",
  "ref_audio_base64": "base64编码的参考音频",
  "num_inference_steps": 50,
  "guidance_scale": 2.0,
  "speed_factor": 1.0,
  "use_sway_sampling": false
}
```

**响应格式**:
```
data: {"audio_chunk": "base64_audio_data", "chunk_index": 0, "is_final": false}
data: {"audio_chunk": "base64_audio_data", "chunk_index": 1, "is_final": false}
data: {"audio_chunk": "base64_audio_data", "chunk_index": 2, "is_final": true, "metadata": {"generation_time": 1.23, "total_chunks": 3}}
```

### 3. 非流式生成
```
POST /v1/generate
```

传统的非流式音频生成端点，返回完整的音频文件。

**请求体**: 与流式端点相同

**响应**:
```json
{
  "audio_base64": "完整的base64编码音频",
  "generation_time": 1.23,
  "duration": 5.67,
  "sample_rate": 24000
}
```

## 使用示例

### 启动服务

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

服务将在 `http://localhost:8000` 启动。

### 使用客户端示例

项目包含一个完整的客户端示例 `streaming_client_example.py`，演示如何使用流式和非流式 API。

```bash
# 准备参考音频文件
# 将你的参考音频文件命名为 reference_audio.wav 并放在项目根目录

# 运行客户端示例
python streaming_client_example.py
```

### 手动测试

#### 1. 健康检查
```bash
curl http://localhost:8000/v1/health
```

#### 2. 流式生成
```bash
curl -X POST "http://localhost:8000/v1/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_text": "Hello, this is a reference text.",
    "gen_text": "This is the text to generate.",
    "ref_audio_base64": "<base64_encoded_audio>",
    "num_inference_steps": 32,
    "guidance_scale": 2.0
  }' \
  --no-buffer
```

#### 3. 非流式生成
```bash
curl -X POST "http://localhost:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_text": "Hello, this is a reference text.",
    "gen_text": "This is the text to generate.",
    "ref_audio_base64": "<base64_encoded_audio>",
    "num_inference_steps": 32,
    "guidance_scale": 2.0
  }'
```

## 配置参数

### 推理参数

- `num_inference_steps`: 推理步数 (默认: 50)
- `guidance_scale`: 引导比例 (默认: 2.0)
- `speed_factor`: 语速因子 (默认: 1.0)
- `use_sway_sampling`: 是否使用 Sway 采样 (默认: false)

### 引擎配置

在 `main.py` 中可以配置以下参数:

- `max_concurrent_requests`: 最大并发请求数
- `chunk_size`: 音频块大小
- `device_mesh`: JAX 设备网格配置

## 性能优化

### 流式传输优势

1. **降低延迟**: 音频块实时传输，无需等待完整生成
2. **内存效率**: 分块处理，减少内存占用
3. **用户体验**: 支持实时播放，提升交互性

### 推理引擎优势

1. **并发处理**: 支持多请求并发处理
2. **资源管理**: 智能的 KV 缓存和内存管理
3. **线程优化**: 分离的预处理、推理和后处理线程

## 故障排除

### 常见问题

1. **服务启动失败**
   - 检查 JAX 设备是否可用
   - 确认模型文件路径正确
   - 查看日志中的错误信息

2. **流式传输中断**
   - 检查网络连接
   - 确认客户端支持 SSE
   - 查看服务器日志

3. **音频质量问题**
   - 调整推理参数
   - 检查参考音频质量
   - 尝试不同的采样设置

### 日志调试

服务使用标准的 Python logging，可以通过设置日志级别来获取详细信息:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 开发说明

### 文件结构

```
jax-F5-TTS-Infer/
├── main.py                    # FastAPI 应用主文件
├── tts_engine.py             # TTS 推理引擎
├── inference_types.py        # 数据类型定义
├── streaming_client_example.py # 客户端示例
├── requirements.txt          # 依赖列表
└── README_STREAMING.md       # 本文档
```

### 扩展开发

1. **自定义音频处理**: 修改 `tts_engine.py` 中的音频编码/解码逻辑
2. **添加新端点**: 在 `main.py` 中添加新的 API 端点
3. **优化性能**: 调整线程数量和缓存策略
4. **监控集成**: 添加指标收集和监控

## 许可证

本项目遵循原 F5-TTS 项目的许可证。