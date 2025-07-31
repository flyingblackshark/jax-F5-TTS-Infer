"""F5-TTS JAX HTTP API server."""

import json
import logging
import time
from typing import Sequence, Optional, Union
from absl import app as abslapp
from absl import flags
from fastapi import APIRouter, Response, HTTPException
import fastapi
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec as P
import jax.numpy as jnp
import flax
from maxdiffusion import pyconfig, max_logging
from maxdiffusion.models.f5.transformers.transformer_f5_flax import F5TextEmbedding, F5Transformer2DModel
from maxdiffusion.max_utils import (
    create_device_mesh,
    get_flash_block_sizes,
    get_precision,
    setup_initial_state,
)
from maxdiffusion.models.modeling_flax_pytorch_utils import convert_f5_state_dict_to_flax
import librosa
from jax_vocos import load_model as load_vocos_model
from maxdiffusion.utils.mel_util import get_mel
from maxdiffusion.utils.pinyin_utils import get_tokenizer, chunk_text, convert_char_to_pinyin, list_str_to_idx
from maxdiffusion.utils.seq_utils import lens_to_mask
import functools
import io
import soundfile as sf
from pydantic import BaseModel
import base64
import tempfile
import os

flags.DEFINE_string("host", "0.0.0.0", "server host address")
flags.DEFINE_integer("port", 8000, "http server port")
flags.DEFINE_string(
    "config",
    "default",
    "configuration name",
)

# --- Configuration & Constants ---
cfg_strength = 2.0
TARGET_SR = 24000
MAX_INFERENCE_STEPS = 100
BUCKET_SIZES = sorted([4, 8, 16, 32, 64])
MAX_CHUNKS = BUCKET_SIZES[-1]

# Global orchestrator for F5-TTS models
f5_orchestrator = None

# --- Pydantic Models for API ---
class TTSRequest(BaseModel):
    ref_text: str
    gen_text: str
    ref_audio_base64: str
    num_inference_steps: int = 50
    guidance_scale: float = 2.0
    speed_factor: float = 1.0
    use_sway_sampling: bool = False

class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration: float
    generation_time: float

# Define Fast API endpoints (use f5_orchestrator to handle).
router = APIRouter()

@router.get("/")
def root():
    """Root path for F5-TTS HTTP Server."""
    return Response(
        content=json.dumps({"message": "F5-TTS HTTP Server"}, indent=4),
        media_type="application/json",
    )

@router.get("/v1/health")
async def health() -> Response:
    """Health check."""
    is_live = f5_orchestrator is not None and f5_orchestrator.is_ready()
    return Response(
        content=json.dumps({"is_live": str(is_live)}, indent=4),
        media_type="application/json",
        status_code=200,
    )

@router.post("/v1/generate", response_model=TTSResponse)
async def generate(request: TTSRequest):
    """Generate TTS audio using JSON request body"""
    start_time = time.perf_counter()
    
    try:
        # Validate ref_text
        if not request.ref_text or not request.ref_text.strip():
            raise HTTPException(status_code=400, detail="ref_text is required and cannot be empty")
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.ref_audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {str(e)}")
        
        # Save to temporary file and load with librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            ref_audio_data, ref_sr = librosa.load(tmp_file_path, sr=None, mono=True)
        finally:
            os.unlink(tmp_file_path)  # Clean up temp file
        
        # Generate audio using orchestrator
        sample_rate, generated_audio, generation_time = f5_orchestrator.generate_audio(
            ref_text=request.ref_text,
            gen_text=request.gen_text,
            ref_audio=ref_audio_data,
            ref_sr=ref_sr,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            speed_factor=request.speed_factor,
            use_sway_sampling=request.use_sway_sampling
        )
        
        # Convert audio to base64
        buffer = io.BytesIO()
        sf.write(buffer, generated_audio, sample_rate, format='WAV')
        audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        duration = len(generated_audio) / sample_rate
        
        return TTSResponse(
            audio_base64=audio_base64,
            sample_rate=sample_rate,
            duration=duration,
            generation_time=generation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        max_logging.error(f"Error during TTS generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/download/{audio_base64}")
async def download_audio(audio_base64: str):
    """Download generated audio file"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=generated_audio.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")


class F5TTSOrchestrator:
    """F5-TTS Model Orchestrator for handling TTS generation."""
    
    def __init__(self, config):
        self.config = config
        self.mesh = None
        self.transformer = None
        self.transformer_state = None
        self.transformer_state_shardings = None
        self.text_encoder = None
        self.text_encoder_params = None
        self.jitted_text_encode_func = None
        self.vocos_model = None
        self.vocos_params = None
        self.jitted_vocos_apply_func = None
        self.vocab_char_map = None
        self.vocab_size = None
        self.p_run_inference_func = None
        self.data_sharding = None
        self.max_sequence_length = None
        self.jitted_get_mel = None
        self._ready = False
        
        self._setup_models_and_state()
    
    def is_ready(self) -> bool:
        """Check if the orchestrator is ready to handle requests."""
        return self._ready
    
    def _setup_models_and_state(self):
        """Setup models and state for F5-TTS."""
        t_start_setup = time.time()
        max_logging.log("Starting one-time setup...")
        
        flash_block_sizes = get_flash_block_sizes(self.config)
        self.max_sequence_length = self.config.max_sequence_length
        max_logging.log(f"Model configured for max sequence length: {self.max_sequence_length}")
        
        rng = jax.random.key(self.config.seed)
        devices_array = create_device_mesh(self.config)
        self.mesh = Mesh(devices_array, self.config.mesh_axes)
        
        if not self.config.mesh_axes:
            raise ValueError("config.mesh_axes must be defined (e.g., ['data'])")
        data_axis_name = self.config.mesh_axes[0]
        model_axis_names = self.config.mesh_axes[1:]
        max_logging.log(f"Using mesh axes: {self.config.mesh_axes} (Data axis: '{data_axis_name}')")
        
        # Define Basic Sharding Specs
        sharding_spec_batch_only = P(data_axis_name)
        sharding_spec_batch_seq = P(data_axis_name, None)
        sharding_spec_batch_seq_dim = P(data_axis_name, None, None)
        sharding_spec_get_mel_input = P(None, data_axis_name)
        sharding_spec_get_mel_output = P(None, data_axis_name, None)
        
        get_mel_in_shardings = (jax.sharding.NamedSharding(self.mesh, sharding_spec_get_mel_input),)
        get_mel_out_shardings = None
        
        self.jitted_get_mel = jax.jit(
            get_mel,
            static_argnums=(1, 2, 3, 4, 5, 6, 8),
            in_shardings=get_mel_in_shardings,
            out_shardings=get_mel_out_shardings
        )
        
        # Load Transformer
        max_logging.log("Loading F5 Transformer model...")
        
        self.transformer = F5Transformer2DModel(
            text_dim=self.config.text_dim,
            mel_dim=self.config.mel_dim,
            dim=self.config.latent_dim,
            head_dim=self.config.head_dim,
            num_depth=self.config.num_depth,
            num_heads=self.config.num_heads,
            mesh=self.mesh,
            attention_kernel=self.config.attention,
            flash_block_sizes=flash_block_sizes,
            dtype=self.config.activations_dtype,
            weights_dtype=self.config.weights_dtype,
            precision=get_precision(self.config),
        )
        
        # Load weights
        transformer_params, text_encoder_params_loaded = convert_f5_state_dict_to_flax(
            self.config.pretrained_model_name_or_path, use_ema=self.config.use_ema
        )
        self.text_encoder_params = flax.core.frozen_dict.FrozenDict(text_encoder_params_loaded)
        
        weights_init_fn = functools.partial(self.transformer.init_weights, rngs=rng, max_sequence_length=self.config.max_sequence_length, eval_only=False)
        self.transformer_state, self.transformer_state_shardings = setup_initial_state(
            model=self.transformer,
            tx=None,
            config=self.config,
            mesh=self.mesh,
            weights_init_fn=weights_init_fn,
            model_params=None,
            training=False,
        )
        self.transformer_state = self.transformer_state.replace(params=transformer_params)
        self.transformer_state = jax.device_put(self.transformer_state, self.transformer_state_shardings)
        
        # Load Text Encoder
        max_logging.log("Loading Text Encoder model...")
        self.vocab_char_map, self.vocab_size = get_tokenizer(self.config.vocab_name_or_path, "custom")
        
        self.text_encoder = F5TextEmbedding(
            precompute_max_pos=self.config.max_sequence_length,
            text_num_embeds=self.config.text_num_embeds,
            text_dim=self.config.text_dim,
            conv_layers=self.config.text_conv_layers,
            dtype=jnp.float32
        )
        
        self.text_encoder_params = jax.device_put(self.text_encoder_params, jax.sharding.NamedSharding(self.mesh, P()))
        max_logging.log("Text encoder params replicated on devices.")
        
        text_encode_in_shardings = (
            jax.sharding.NamedSharding(self.mesh, P()),
            jax.sharding.NamedSharding(self.mesh, sharding_spec_batch_seq),
            jax.sharding.NamedSharding(self.mesh, sharding_spec_batch_seq),
            jax.sharding.NamedSharding(self.mesh, P()),
        )
        text_encode_out_shardings = jax.sharding.NamedSharding(self.mesh, sharding_spec_batch_seq_dim)
        
        def wrap_text_encoder_apply(params, text_ids, text_decoder_segment_ids, rngs):
            return self.text_encoder.apply(params, text_ids, text_decoder_segment_ids, rngs=rngs)
        
        self.jitted_text_encode_func = jax.jit(
            wrap_text_encoder_apply,
            in_shardings=text_encode_in_shardings,
            out_shardings=text_encode_out_shardings,
            static_argnums=()
        )
        
        max_logging.log("Text Encoder JIT created.")
        
        # Load Vocoder
        max_logging.log("Loading Vocoder model...")
        self.vocos_model, vocos_params_loaded = load_vocos_model(self.config.vocoder_model_path)
        self.vocos_params = flax.core.frozen_dict.FrozenDict(vocos_params_loaded)
        
        self.vocos_params = jax.device_put(self.vocos_params, jax.sharding.NamedSharding(self.mesh, P()))
        max_logging.log("Vocoder params replicated on devices.")
        
        vocos_apply_in_shardings = (
            jax.sharding.NamedSharding(self.mesh, P()),
            jax.sharding.NamedSharding(self.mesh, sharding_spec_batch_seq_dim),
            jax.sharding.NamedSharding(self.mesh, P()),
        )
        
        vocos_apply_out_shardings = jax.sharding.NamedSharding(self.mesh, sharding_spec_batch_seq)
        
        def wrap_vocos_apply(params, x, rngs):
            return self.vocos_model.apply(params, x, rngs=rngs)
        
        self.jitted_vocos_apply_func = jax.jit(
            wrap_vocos_apply,
            in_shardings=vocos_apply_in_shardings,
            out_shardings=vocos_apply_out_shardings,
            static_argnums=()
        )
        
        self.data_sharding = jax.sharding.NamedSharding(self.mesh, P(self.config.data_sharding[0]))
        
        # Define shardings for inputs to run_inference
        latents_sharding = self.data_sharding
        cond_sharding = self.data_sharding
        decoder_segment_ids_sharding = self.data_sharding
        text_embed_sharding = self.data_sharding
        ts_sharding = jax.sharding.NamedSharding(self.mesh, P())
        
        # JIT the run_inference function
        partial_run_inference = functools.partial(
            self._run_inference,
            transformer=self.transformer,
            config=self.config,
            mesh=self.mesh,
        )
        
        in_shardings_inf = (
            self.transformer_state_shardings,
            latents_sharding,
            cond_sharding,
            decoder_segment_ids_sharding,
            text_embed_sharding,
            text_embed_sharding,
            ts_sharding,
            ts_sharding
        )
        out_shardings_inf = latents_sharding
        self.p_run_inference_func = jax.jit(
            partial_run_inference,
            static_argnums=(),
            in_shardings=in_shardings_inf,
            out_shardings=out_shardings_inf,
        )
        
        t_end_setup = time.time()
        max_logging.log(f"Setup completed in {t_end_setup - t_start_setup:.2f}s")
        self._ready = True
    
    def _loop_body(self, step, args, transformer, cond, decoder_segment_ids, text_embed_cond, text_embed_uncond):
        """Core diffusion loop body."""
        latents, state, c_ts, p_ts = args
        latents_dtype = latents.dtype
        t_curr = c_ts[step]
        t_prev = p_ts[step]
        t_vec = jnp.full((latents.shape[0],), t_curr, dtype=latents.dtype)
        
        # Conditional prediction
        pred = transformer.apply(
            {"params": state.params},
            x=latents,
            cond=cond,
            decoder_segment_ids=decoder_segment_ids,
            text_embed=text_embed_cond,
            timestep=t_vec,
        )
        
        # Unconditional prediction
        pred_uncond = transformer.apply(
            {"params": state.params},
            x=latents,
            cond=cond,
            decoder_segment_ids=decoder_segment_ids,
            text_embed=text_embed_uncond,
            timestep=t_vec,
        )
        
        # Classifier-free guidance
        pred = pred_uncond + cfg_strength * (pred - pred_uncond)
        
        # Euler step
        latents = latents + (t_prev - t_curr) * pred
        
        return (latents, state, c_ts, p_ts)
    
    def _run_inference(self, states, latents, cond, decoder_segment_ids, text_embed_cond, text_embed_uncond, c_ts, p_ts, transformer, config, mesh):
        """Run the diffusion inference loop."""
        num_steps = len(c_ts)
        
        def scan_fn(carry, step):
            return self._loop_body(step, carry, transformer, cond, decoder_segment_ids, text_embed_cond, text_embed_uncond), None
        
        final_carry, _ = jax.lax.scan(scan_fn, (latents, states, c_ts, p_ts), jnp.arange(num_steps))
        final_latents, _, _, _ = final_carry
        
        return final_latents
    
    def generate_audio(self, ref_text: str, gen_text: str, ref_audio: np.ndarray, ref_sr: int, 
                      num_inference_steps: int = 50, guidance_scale: float = 2.0, 
                      speed_factor: float = 1.0, use_sway_sampling: bool = False) -> tuple[int, np.ndarray, float]:
        """Generate TTS audio."""
        if not self._ready:
            raise RuntimeError("Orchestrator is not ready")
        
        start_time = time.time()
        
        # Validate ref_text
        if not ref_text or not ref_text.strip():
            raise HTTPException(status_code=400, detail="ref_text is required and cannot be empty")
        
        # Process reference audio
        if ref_sr != TARGET_SR:
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=TARGET_SR)
        
        # Convert text to tokens
        ref_text_list = [ref_text]
        gen_text_list = [gen_text]
        
        ref_text_tokens = []
        gen_text_tokens = []
        
        for text in ref_text_list:
            text_tokens = convert_char_to_pinyin([text], self.vocab_char_map)
            ref_text_tokens.extend(text_tokens)
        
        for text in gen_text_list:
            text_tokens = convert_char_to_pinyin([text], self.vocab_char_map)
            gen_text_tokens.extend(text_tokens)
        
        # Prepare inputs
        ref_audio_mel = self.jitted_get_mel(
            ref_audio[None, :], TARGET_SR, 1024, 256, 1024, 0.0, None, True, "reflect"
        )[0]
        
        ref_audio_len = ref_audio_mel.shape[-1]
        gen_audio_len = int(ref_audio_len * len(gen_text) / len(ref_text) * speed_factor)
        
        # Create latents
        latents_shape = (1, ref_audio_mel.shape[0], ref_audio_len + gen_audio_len)
        latents = jax.random.normal(jax.random.key(42), latents_shape, dtype=jnp.float32)
        
        # Prepare conditioning
        cond = jnp.concatenate([ref_audio_mel, jnp.zeros((ref_audio_mel.shape[0], gen_audio_len))], axis=-1)[None, :, :]
        
        # Create segment IDs
        decoder_segment_ids = jnp.concatenate([
            jnp.ones((1, ref_audio_len), dtype=jnp.int32),
            jnp.zeros((1, gen_audio_len), dtype=jnp.int32)
        ], axis=-1)
        
        # Encode text
        all_text_tokens = ref_text_tokens + gen_text_tokens
        text_ids = jnp.array([list_str_to_idx(all_text_tokens, self.vocab_char_map)])
        text_decoder_segment_ids = jnp.concatenate([
            jnp.ones((1, len(ref_text_tokens)), dtype=jnp.int32),
            jnp.zeros((1, len(gen_text_tokens)), dtype=jnp.int32)
        ], axis=-1)
        
        rng = jax.random.key(42)
        text_embed_cond = self.jitted_text_encode_func(
            {"params": self.text_encoder_params}, text_ids, text_decoder_segment_ids, rng
        )
        
        # Create unconditional text embedding (empty text)
        empty_text_ids = jnp.zeros_like(text_ids)
        text_embed_uncond = self.jitted_text_encode_func(
            {"params": self.text_encoder_params}, empty_text_ids, text_decoder_segment_ids, rng
        )
        
        # Create timesteps
        if use_sway_sampling:
            # Implement sway sampling if needed
            c_ts = jnp.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            p_ts = jnp.linspace(1.0, 0.0, num_inference_steps + 1)[1:]
        else:
            c_ts = jnp.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            p_ts = jnp.linspace(1.0, 0.0, num_inference_steps + 1)[1:]
        
        # Run inference
        final_latents = self.p_run_inference_func(
            self.transformer_state, latents, cond, decoder_segment_ids, 
            text_embed_cond, text_embed_uncond, c_ts, p_ts
        )
        
        # Decode with vocoder
        rng = jax.random.key(42)
        generated_audio = self.jitted_vocos_apply_func(
            {"params": self.vocos_params}, final_latents, rng
        )
        
        # Convert to numpy and extract generated portion
        generated_audio = np.array(generated_audio[0, ref_audio_len:])
        
        generation_time = time.time() - start_time
        
        return TARGET_SR, generated_audio, generation_time


def server(argv: Sequence[str]):
    """Main server function."""
    # Init Fast API.
    app = fastapi.FastAPI(title="F5-TTS HTTP Server", version="1.0.0")
    app.include_router(router)
    
    # Initialize pyconfig with command line arguments
    pyconfig.initialize(argv)
    config = pyconfig.config
    
    print(f"Server config: {config}")
    del argv
    
    global f5_orchestrator
    f5_orchestrator = F5TTSOrchestrator(config=config)
    
    # Start uvicorn http server.
    uvicorn.run(
        app, host=flags.FLAGS.host, port=flags.FLAGS.port, log_level="info"
    )


if __name__ == "__main__":
    # Run Abseil app w flags parser.
    abslapp.run(server)