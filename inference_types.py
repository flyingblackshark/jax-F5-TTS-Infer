"""Request and response types for F5-TTS inference engine."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import jax


@dataclass
class TTSResponse:
    """Response containing generated audio chunk or metadata."""
    audio_chunk: Optional[bytes] = None  # Audio chunk in bytes
    sample_rate: Optional[int] = None
    is_final: bool = False  # Whether this is the final chunk
    metadata: Optional[dict] = None  # Additional metadata


@dataclass
class OnlineTTSRequest:
    """Online TTS request for streaming."""
    ref_text: str
    gen_text: str
    ref_audio_base64: str
    num_inference_steps: int = 50
    guidance_scale: float = 2.0
    speed_factor: float = 1.0
    use_sway_sampling: bool = False
    res_queue: asyncio.Queue[TTSResponse] = field(default_factory=asyncio.Queue)


@dataclass
class OfflineTTSRequest:
    """Offline TTS request for batch processing."""
    ref_text: str
    gen_text: str
    ref_audio_base64: str
    num_inference_steps: int = 50
    guidance_scale: float = 2.0
    speed_factor: float = 1.0
    use_sway_sampling: bool = False


@dataclass
class TTSInferenceRequest:
    """Internal request for TTS inference processing."""
    request_id: str
    ref_text: str
    gen_text: str
    ref_audio: np.ndarray
    ref_sr: int
    num_inference_steps: int
    guidance_scale: float
    speed_factor: float
    use_sway_sampling: bool
    response_queue: Optional[asyncio.Queue[TTSResponse]] = None
    timestamp: Optional[float] = None


@dataclass
class TTSProcessRequest:
    """Request for TTS processing pipeline."""
    id: str
    ref_audio: np.ndarray
    ref_sr: int
    ref_text: str
    gen_text: str
    num_inference_steps: int
    guidance_scale: float
    speed_factor: float
    use_sway_sampling: bool
    device_mel: jax.Array
    device_text_embed: jax.Array
    chunk_size: int


@dataclass
class TTSPostProcessRequest:
    """Post-process request for TTS output."""
    request_id: str
    audio_chunks: list[np.ndarray]
    sample_rate: int
    is_final: bool = False
    generation_time: float = 0.0