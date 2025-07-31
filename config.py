"""Configuration file for F5-TTS FastAPI service"""

import os
from dataclasses import dataclass
from typing import List, Optional
import jax.numpy as jnp

@dataclass
class F5TTSConfig:
    # Model paths
    pretrained_model_name_or_path: str = "F5-TTS"
    vocoder_model_path: str = "charactr/vocos-mel-24khz"
    vocab_name_or_path: str = "Emilia_ZH_EN"
    
    # Model architecture
    text_dim: int = 512
    mel_dim: int = 100
    latent_dim: int = 1024
    head_dim: int = 64
    num_depth: int = 22
    num_heads: int = 16
    text_num_embeds: int = 256
    text_conv_layers: int = 0
    
    # Training/inference settings
    max_sequence_length: int = 4096
    seed: int = 42
    use_ema: bool = True
    
    # JAX/Flax settings
    mesh_axes: List[str] = None
    data_sharding: List[str] = None
    activations_dtype: str = "float32"
    weights_dtype: str = "float32"
    precision: str = "default"
    attention: str = "flash"
    
    # Bucket sizes for batching
    bucket_sizes: List[int] = None
    
    # Sway sampling
    sway_sampling_coef: Optional[float] = 0.0
    
    def __post_init__(self):
        if self.mesh_axes is None:
            self.mesh_axes = ["data"]
        if self.data_sharding is None:
            self.data_sharding = ["data"]
        if self.bucket_sizes is None:
            self.bucket_sizes = [4, 8, 16, 32, 64]

# Default configuration
default_config = F5TTSConfig()

# Environment variable overrides
if os.getenv("F5_MODEL_PATH"):
    default_config.pretrained_model_name_or_path = os.getenv("F5_MODEL_PATH")
if os.getenv("VOCODER_MODEL_PATH"):
    default_config.vocoder_model_path = os.getenv("VOCODER_MODEL_PATH")
if os.getenv("VOCAB_PATH"):
    default_config.vocab_name_or_path = os.getenv("VOCAB_PATH")
if os.getenv("MAX_SEQUENCE_LENGTH"):
    default_config.max_sequence_length = int(os.getenv("MAX_SEQUENCE_LENGTH"))