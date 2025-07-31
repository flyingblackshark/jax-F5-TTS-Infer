from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from typing import Optional, Union
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
import time
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

# --- Configuration & Constants ---
cfg_strength = 2.0
TARGET_SR = 24000
MAX_INFERENCE_STEPS = 100
BUCKET_SIZES = sorted([4, 8, 16, 32, 64])
MAX_CHUNKS = BUCKET_SIZES[-1]

# --- Global Variables for Model State ---
global_config = None
global_mesh = None
global_transformer = None
global_transformer_state = None
global_transformer_state_shardings = None
global_text_encoder = None
global_text_encoder_params = None
global_jitted_text_encode_func = None
global_vocos_model = None
global_vocos_params = None
global_jitted_vocos_apply_func = None
global_vocab_char_map = None
global_vocab_size = None
global_p_run_inference_func = None
global_data_sharding = None
global_max_sequence_length = None
jitted_get_mel = None

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

# --- Core Diffusion Loop Logic ---
def loop_body(
    step,
    args,
    transformer,
    cond,
    decoder_segment_ids,
    text_embed_cond,
    text_embed_uncond,
):
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
    null_pred = transformer.apply(
        {"params": state.params},
        x=latents,
        cond=jnp.zeros_like(cond),
        decoder_segment_ids=decoder_segment_ids,
        text_embed=text_embed_uncond,
        timestep=t_vec,
    )

    # Classifier-Free Guidance
    guidance_scale = cfg_strength
    pred = null_pred + guidance_scale * (pred - null_pred)

    # DDIM-like step (simplified Euler)
    latents = latents + (t_prev - t_curr) * pred
    latents = jnp.array(latents, dtype=latents_dtype)

    return latents, state, c_ts, p_ts

def run_inference(
    states, latents, cond, decoder_segment_ids, text_embed_cond, text_embed_uncond, c_ts, p_ts, transformer, config, mesh
):
    transformer_state = states

    loop_body_p = functools.partial(
        loop_body,
        transformer=transformer,
        cond=cond,
        decoder_segment_ids=decoder_segment_ids,
        text_embed_cond=text_embed_cond,
        text_embed_uncond=text_embed_uncond,
    )

    latents_final, _, _, _ = jax.lax.fori_loop(0, len(c_ts), loop_body_p, (latents, transformer_state, c_ts, p_ts))

    return latents_final

# --- Audio Generation Function ---
def generate_audio_api(
    ref_text: str,
    gen_text: str,
    ref_audio: np.ndarray,
    ref_sr: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.0,
    speed_factor: float = 1.0,
    use_sway_sampling: bool = False,
) -> tuple[int, np.ndarray, float]:
    """
    Main function for API audio generation.
    Returns: (sample_rate, audio_array, generation_time)
    """
    global cfg_strength
    cfg_strength = guidance_scale

    t_start_total = time.time()
    max_logging.log(f"Starting audio generation... Steps: {num_inference_steps}, CFG: {guidance_scale}, Speed: {speed_factor}, Sway: {use_sway_sampling}")

    # --- Input Validation and Loading ---
    if not ref_text:
        raise HTTPException(status_code=400, detail="Reference text cannot be empty.")
    if not gen_text:
        raise HTTPException(status_code=400, detail="Generation text cannot be empty.")
    if ref_audio is None or ref_audio.size == 0:
        raise HTTPException(status_code=400, detail="Reference audio is required.")

    # Process reference audio
    if ref_sr != TARGET_SR:
        max_logging.log(f"Resampling reference audio from {ref_sr} Hz to {TARGET_SR} Hz.")
        ref_audio = librosa.resample(ref_audio.astype(np.float32), orig_sr=ref_sr, target_sr=TARGET_SR)
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=1)  # Ensure mono
    max_logging.log("Loaded reference audio from API input.")

    if ref_audio.size == 0:
        raise HTTPException(status_code=400, detail="Reference audio is empty after loading.")

    # --- Preprocessing ---
    t_start_preprocess = time.time()
    max_logging.log("Preprocessing text and audio...")

    # Ensure reference text ends with space if last char is ASCII
    if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    # Estimate character count per second from reference
    ref_duration_sec = len(ref_audio) / TARGET_SR
    if ref_duration_sec < 0.1:
        raise HTTPException(status_code=400, detail="Reference audio is too short (must be at least 0.1 seconds).")

    # Calculate max characters for chunking based on reference speech rate
    chars_per_sec_ref = len(ref_text.encode("utf-8")) / ref_duration_sec
    max_gen_duration_sec = global_max_sequence_length * 256 / TARGET_SR - ref_duration_sec
    if max_gen_duration_sec <= 0:
        raise HTTPException(status_code=400, detail=f"Reference audio duration ({ref_duration_sec:.1f}s) exceeds max allowed duration ({global_max_sequence_length * 256 / TARGET_SR}s).")

    estimated_max_chars = max(10, int(chars_per_sec_ref * max_gen_duration_sec * 0.8 * speed_factor))
    max_logging.log(f"Reference: {ref_duration_sec:.1f}s, {len(ref_text)} chars. Estimated max chars/chunk: {estimated_max_chars}")

    gen_text_batches = chunk_text(gen_text, max_chars=estimated_max_chars)
    num_chunks = len(gen_text_batches)
    max_logging.log(f"Split generation text into {num_chunks} chunks.")

    MAX_CHUNKS = global_config.bucket_sizes[-1]
    BUCKET_SIZES = global_config.bucket_sizes
    
    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="Text processing resulted in zero valid chunks. Try different text.")
    if num_chunks > MAX_CHUNKS:
        raise HTTPException(status_code=400, detail=f"Too many text chunks ({num_chunks}). Maximum allowed is {MAX_CHUNKS}. Please shorten the 'Text to Generate'.")

    # Find the target batch size from buckets
    target_batch_size = MAX_CHUNKS
    for bucket in BUCKET_SIZES:
        if num_chunks <= bucket:
            target_batch_size = bucket
            break
    
    padded_items_count = target_batch_size - num_chunks
    total_batch_items = target_batch_size

    max_logging.log(f"Processing {num_chunks} chunks. Padding to nearest bucket size: {target_batch_size} (adding {padded_items_count} padding items).")

    batched_text_list_combined = []
    batched_duration_frames = []
    hop_length = 256
    ref_audio_len_frames = ref_audio.shape[-1] // hop_length + 1

    # Limit reference audio / text to avoid exceeding max sequence length early
    max_ref_frames = int(global_max_sequence_length * 0.6)
    if ref_audio_len_frames > max_ref_frames:
        max_logging.log(f"Warning: Truncating reference audio from {ref_audio_len_frames} to {max_ref_frames} frames.")
        ref_audio_len_frames = max_ref_frames
        ref_audio = ref_audio[:ref_audio_len_frames * hop_length]
        original_ref_text_len = len(ref_text)
        ref_text = ref_text[:int(original_ref_text_len * (max_ref_frames / (ref_audio.shape[-1] // hop_length + 1)))]
        if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
            ref_text += " "
        max_logging.log(f"Truncated reference text length: {len(ref_text)}")

    if ref_audio_len_frames >= global_max_sequence_length:
        raise HTTPException(status_code=400, detail=f"Reference audio ({ref_audio_len_frames} frames) already exceeds max sequence length ({global_max_sequence_length}). Please use shorter audio.")

    for i, single_gen_text in enumerate(gen_text_batches):
        text_combined = ref_text + single_gen_text
        batched_text_list_combined.append(text_combined)

        ref_text_byte_len = len(ref_text.encode('utf-8'))
        gen_text_byte_len = len(single_gen_text.encode('utf-8'))

        if ref_text_byte_len > 0:
            estimated_gen_frames = int(ref_audio_len_frames / ref_text_byte_len * gen_text_byte_len / speed_factor)
        else:
            avg_chars_per_sec = 5 * speed_factor
            estimated_gen_frames = int(gen_text_byte_len * (TARGET_SR / hop_length) / avg_chars_per_sec) if avg_chars_per_sec > 0 else 50

        estimated_gen_frames = max(0, estimated_gen_frames)
        duration_frames = ref_audio_len_frames + estimated_gen_frames
        duration_frames = min(global_max_sequence_length, duration_frames)
        duration_frames = max(ref_audio_len_frames + 1, duration_frames)

        batched_duration_frames.append(duration_frames)
        max_logging.log(f"Chunk {i+1}/{len(gen_text_batches)}: Combined text len: {len(text_combined)}, Estimated total frames: {duration_frames}")

    # Convert text to pinyin/chars list
    pinyin_start_time = time.time()
    final_text_list_pinyin = convert_char_to_pinyin(batched_text_list_combined)
    max_logging.log(f"Pinyin conversion took {time.time() - pinyin_start_time:.2f}s")

    text_ids_unpadded = list_str_to_idx(final_text_list_pinyin, global_vocab_char_map, max_length=global_max_sequence_length)
    text_ids = np.pad(text_ids_unpadded, ((0, padded_items_count), (0, 0)), constant_values=0)
    ref_audio_padded = np.pad(ref_audio, (0, max(0, global_max_sequence_length * hop_length + hop_length - ref_audio.shape[0])))
    ref_audio_padded = ref_audio_padded[np.newaxis, :]
    cond = jitted_get_mel(ref_audio_padded)
    cond_pad_len = global_max_sequence_length - cond.shape[1]
    if cond_pad_len > 0:
        cond = np.pad(cond, ((0,0), (0, cond_pad_len), (0,0)))
    elif cond_pad_len < 0:
        cond = cond[:, :global_max_sequence_length, :]
    cond = np.repeat(cond, total_batch_items, axis=0)

    safe_padding_duration = ref_audio_len_frames + 1
    padded_durations = batched_duration_frames + [safe_padding_duration] * padded_items_count
    duration_frames_arr = np.array(padded_durations, dtype=np.int32)

    ref_len_frames_arr = np.array([ref_audio_len_frames] * total_batch_items, dtype=np.int32)
    duration_frames_arr = np.minimum(duration_frames_arr, global_max_sequence_length)
    duration_frames_arr = np.maximum(duration_frames_arr, ref_len_frames_arr + 1)

    text_lens = np.minimum((text_ids != 0).sum(axis=-1), global_max_sequence_length)
    effective_min_len = np.maximum(text_lens, ref_len_frames_arr) + 1
    duration_final = np.maximum(effective_min_len, duration_frames_arr)
    duration_final = np.minimum(duration_final, global_max_sequence_length)

    cond_mask = lens_to_mask(ref_len_frames_arr, length=global_max_sequence_length)
    decoder_mask = lens_to_mask(duration_final, length=global_max_sequence_length)

    text_decoder_segment_ids = (text_ids != 0).astype(np.int32)
    decoder_segment_ids = decoder_mask.astype(np.int32)

    step_cond = np.where(cond_mask[..., np.newaxis], cond, np.zeros_like(cond))

    # Shard data
    step_cond = jax.device_put(step_cond, global_data_sharding)
    text_ids = jax.device_put(text_ids, global_data_sharding)
    decoder_segment_ids = jax.device_put(decoder_segment_ids, global_data_sharding)
    text_decoder_segment_ids = jax.device_put(text_decoder_segment_ids, global_data_sharding)
    cond_mask_sharded = jax.device_put(cond_mask, global_data_sharding)

    t_end_preprocess = time.time()
    max_logging.log(f"Preprocessing finished in {t_end_preprocess - t_start_preprocess:.2f}s.")

    # --- Text Embedding ---
    t_start_embed = time.time()
    max_logging.log("Generating text embeddings...")
    rng_embed = jax.random.key(global_config.seed + 1)
    rngs_embed = {'params': rng_embed, 'dropout': rng_embed}

    text_embed_cond = global_jitted_text_encode_func({"params": global_text_encoder_params},
                                          text_ids,
                                          text_decoder_segment_ids,
                                         rngs_embed)

    text_embed_uncond = global_jitted_text_encode_func({"params": global_text_encoder_params},
                                  np.zeros_like(text_ids),
                                  text_decoder_segment_ids,
                                  rngs_embed)
    t_end_embed = time.time()
    max_logging.log(f"Text embedding generation took {t_end_embed - t_start_embed:.2f}s.")

    # --- Diffusion Sampling ---
    t_start_diffusion = time.time()
    max_logging.log(f"Starting diffusion sampling with {num_inference_steps} steps...")

    latents_shape = (total_batch_items, global_max_sequence_length, 100)
    latents_rng = jax.random.key(global_config.seed + 2)
    latents = jax.random.normal(latents_rng, latents_shape, dtype=jnp.float32)
    latents = jax.device_put(latents, global_data_sharding)

    t_start = 0.0
    timesteps = np.linspace(t_start, 1.0, num_inference_steps + 1).astype(np.float32)

    if use_sway_sampling:
        sway_coef = global_config.sway_sampling_coef
        if sway_coef is not None:
            max_logging.log(f"Applying Sway Sampling with coefficient: {sway_coef}")
            timesteps = timesteps + sway_coef * (np.cos(np.pi / 2 * timesteps) - 1 + timesteps)
            timesteps = np.clip(timesteps, 0.0, 1.0)
        else:
            max_logging.log("Sway sampling enabled but coefficient is 0 or missing in config. Skipping.")
    else:
        max_logging.log("Sway sampling disabled.")

    c_ts = timesteps[:-1]
    p_ts = timesteps[1:]

    y_final_latents = global_p_run_inference_func(
        global_transformer_state,
        latents,
        step_cond,
        decoder_segment_ids,
        text_embed_cond,
        text_embed_uncond,
        c_ts,
        p_ts
    )

    y_final_latents.block_until_ready()
    t_end_diffusion = time.time()
    max_logging.log(f"Diffusion sampling finished in {t_end_diffusion - t_start_diffusion:.2f}s.")

    # --- Postprocessing (Vocoder) ---
    t_start_post = time.time()
    max_logging.log("Applying Vocoder...")

    out_latents = jnp.where(cond_mask_sharded[..., jnp.newaxis], cond, y_final_latents)

    vocoder_rng = jax.random.key(global_config.seed + 3)
    rngs_vocoder = {'params': vocoder_rng, 'dropout': vocoder_rng}
    audio_out_jax = global_jitted_vocos_apply_func({"params": global_vocos_params}, out_latents, rngs_vocoder)
    audio_out_jax.block_until_ready()

    max_logging.log("Transferring generated audio to CPU...")
    cpu_durations = np.array(batched_duration_frames)
    cpu_ref_len_frames = ref_audio_len_frames
    audio_out_cpu = np.asarray(audio_out_jax[:num_chunks])

    t_end_post = time.time()
    max_logging.log(f"Vocoder and transfer took {t_end_post - t_start_post:.2f}s.")

    # --- Final Audio Stitching ---
    t_start_stitch = time.time()
    max_logging.log("Stitching audio chunks...")
    final_audio_segments = []

    ref_len_samples = cpu_ref_len_frames * hop_length

    for i in range(num_chunks):
        current_duration_frames = cpu_durations[i]
        current_duration_samples = current_duration_frames * hop_length
        generated_part = audio_out_cpu[i, ref_len_samples:current_duration_samples]
        final_audio_segments.append(generated_part)

    final_audio = np.concatenate(final_audio_segments) if final_audio_segments else np.array([], dtype=np.float32)

    t_end_stitch = time.time()
    max_logging.log(f"Audio stitching took {t_end_stitch - t_start_stitch:.2f}s.")

    t_end_total = time.time()
    total_duration = t_end_total - t_start_total
    generated_audio_duration = len(final_audio) / TARGET_SR
    max_logging.log(f"Total generation time: {total_duration:.2f}s for {generated_audio_duration:.2f}s of audio.")
    if generated_audio_duration > 0:
        rtf = total_duration / generated_audio_duration
        max_logging.log(f"Real-Time Factor (RTF): {rtf:.3f}")

    return (TARGET_SR, final_audio, total_duration)

# --- Setup Function ---
def setup_models_and_state(config):
    global global_config, global_mesh, global_transformer, global_transformer_state
    global global_transformer_state_shardings, global_text_encoder, global_text_encoder_params
    global global_jitted_text_encode_func, global_vocos_model, global_vocos_params
    global global_jitted_vocos_apply_func, global_vocab_char_map, global_vocab_size
    global global_p_run_inference_func, global_data_sharding, global_max_sequence_length
    global jitted_get_mel

    t_start_setup = time.time()
    max_logging.log("Starting one-time setup...")
    global_config = config

    flash_block_sizes = get_flash_block_sizes(config)
    global_max_sequence_length = config.max_sequence_length
    max_logging.log(f"Model configured for max sequence length: {global_max_sequence_length}")

    rng = jax.random.key(config.seed)
    devices_array = create_device_mesh(config)
    global_mesh = Mesh(devices_array, config.mesh_axes)
    mesh = global_mesh

    if not config.mesh_axes:
        raise ValueError("config.mesh_axes must be defined (e.g., ['data'])")
    data_axis_name = config.mesh_axes[0]
    model_axis_names = config.mesh_axes[1:]
    max_logging.log(f"Using mesh axes: {config.mesh_axes} (Data axis: '{data_axis_name}')")

    # Define Basic Sharding Specs
    sharding_spec_batch_only = P(data_axis_name)
    sharding_spec_batch_seq = P(data_axis_name, None)
    sharding_spec_batch_seq_dim = P(data_axis_name, None, None)
    sharding_spec_get_mel_input = P(None, data_axis_name)
    sharding_spec_get_mel_output = P(None, data_axis_name, None)

    get_mel_in_shardings = (jax.sharding.NamedSharding(mesh, sharding_spec_get_mel_input),)
    get_mel_out_shardings = None

    jitted_get_mel = jax.jit(
        get_mel,
        static_argnums=(1, 2, 3, 4, 5, 6, 8),
        in_shardings=get_mel_in_shardings,
        out_shardings=get_mel_out_shardings
    )

    # Load Transformer
    max_logging.log("Loading F5 Transformer model...")

    global_transformer = F5Transformer2DModel(
        text_dim=config.text_dim,
        mel_dim=config.mel_dim,
        dim=config.latent_dim,
        head_dim=config.head_dim,
        num_depth=config.num_depth,
        num_heads=config.num_heads,
        mesh=mesh,
        attention_kernel=config.attention,
        flash_block_sizes=flash_block_sizes,
        dtype=config.activations_dtype,
        weights_dtype=config.weights_dtype,
        precision=get_precision(config),
    )
    transformer = global_transformer

    # Load weights
    transformer_params, text_encoder_params_loaded = convert_f5_state_dict_to_flax(
        config.pretrained_model_name_or_path, use_ema=config.use_ema
    )
    global_text_encoder_params = flax.core.frozen_dict.FrozenDict(text_encoder_params_loaded)

    weights_init_fn = functools.partial(transformer.init_weights, rngs=rng, max_sequence_length=config.max_sequence_length, eval_only=False)
    global_transformer_state, global_transformer_state_shardings = setup_initial_state(
        model=transformer,
        tx=None,
        config=config,
        mesh=mesh,
        weights_init_fn=weights_init_fn,
        model_params=None,
        training=False,
    )
    global_transformer_state = global_transformer_state.replace(params=transformer_params)
    global_transformer_state = jax.device_put(global_transformer_state, global_transformer_state_shardings)

    # Load Text Encoder
    max_logging.log("Loading Text Encoder model...")
    global_vocab_char_map, global_vocab_size = get_tokenizer(config.vocab_name_or_path, "custom")

    global_text_encoder = F5TextEmbedding(
        precompute_max_pos=config.max_sequence_length,
        text_num_embeds=config.text_num_embeds,
        text_dim=config.text_dim,
        conv_layers=config.text_conv_layers,
        dtype=jnp.float32
    )

    global_text_encoder_params = jax.device_put(global_text_encoder_params, jax.sharding.NamedSharding(mesh, P()))
    max_logging.log("Text encoder params replicated on devices.")

    text_encode_in_shardings = (
        jax.sharding.NamedSharding(mesh, P()),
        jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq),
        jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq),
        jax.sharding.NamedSharding(mesh, P()),
    )
    text_encode_out_shardings = jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq_dim)
    
    def wrap_text_encoder_apply(params, text_ids, text_decoder_segment_ids, rngs):
        return global_text_encoder.apply(params, text_ids, text_decoder_segment_ids, rngs=rngs)

    global_jitted_text_encode_func = jax.jit(
        wrap_text_encoder_apply,
        in_shardings=text_encode_in_shardings,
        out_shardings=text_encode_out_shardings,
        static_argnums=()
    )

    max_logging.log("Text Encoder JIT created.")

    # Load Vocoder
    max_logging.log("Loading Vocoder model...")
    global_vocos_model, vocos_params_loaded = load_vocos_model(config.vocoder_model_path)
    global_vocos_params = flax.core.frozen_dict.FrozenDict(vocos_params_loaded)

    global_vocos_params = jax.device_put(global_vocos_params, jax.sharding.NamedSharding(mesh, P()))
    max_logging.log("Vocoder params replicated on devices.")

    vocos_apply_in_shardings = (
        jax.sharding.NamedSharding(mesh, P()),
        jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq_dim),
        jax.sharding.NamedSharding(mesh, P()),
    )

    vocos_apply_out_shardings = jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq)
    
    def wrap_vocos_apply(params, x, rngs):
        return global_vocos_model.apply(params, x, rngs=rngs)

    global_jitted_vocos_apply_func = jax.jit(
        wrap_vocos_apply,
        in_shardings=vocos_apply_in_shardings,
        out_shardings=vocos_apply_out_shardings,
        static_argnums=()
    )

    global_data_sharding = jax.sharding.NamedSharding(mesh, P(config.data_sharding[0]))

    # Define shardings for inputs to run_inference
    latents_sharding = global_data_sharding
    cond_sharding = global_data_sharding
    decoder_segment_ids_sharding = global_data_sharding
    text_embed_sharding = global_data_sharding
    ts_sharding = jax.sharding.NamedSharding(mesh, P())

    # JIT the run_inference function
    partial_run_inference = functools.partial(
        run_inference,
        transformer=transformer,
        config=config,
        mesh=mesh,
    )

    in_shardings_inf = (
        global_transformer_state_shardings,
        latents_sharding,
        cond_sharding,
        decoder_segment_ids_sharding,
        text_embed_sharding,
        text_embed_sharding,
        ts_sharding,
        ts_sharding
    )
    out_shardings_inf = latents_sharding
    global_p_run_inference_func = jax.jit(
        partial_run_inference,
        static_argnums=(),
        in_shardings=in_shardings_inf,
        out_shardings=out_shardings_inf,
    )

    t_end_setup = time.time()
    max_logging.log(f"Setup completed in {t_end_setup - t_start_setup:.2f}s")

# --- FastAPI Application ---
app = FastAPI(title="F5-TTS Inference API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        # Initialize pyconfig with default arguments
        import sys
        pyconfig.initialize(sys.argv)
        config = pyconfig.config
        
        # Perform one-time setup
        setup_models_and_state(config)
        max_logging.log("FastAPI server started successfully")
    except Exception as e:
        max_logging.error(f"Fatal error during startup: {e}", exc_info=True)
        raise e

@app.get("/")
async def root():
    return {"message": "F5-TTS Inference API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": global_transformer is not None}

@app.post("/generate", response_model=TTSResponse)
async def generate_tts(request: TTSRequest):
    """Generate TTS audio using JSON request body"""
    try:
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
        
        # Generate audio
        sample_rate, generated_audio, generation_time = generate_audio_api(
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

@app.get("/download/{audio_base64}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)