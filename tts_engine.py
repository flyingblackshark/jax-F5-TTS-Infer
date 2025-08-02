"""F5-TTS Inference Engine based on JetStream architecture."""

import enum
import dataclasses
import datetime
import queue
import threading
import time
import uuid
from typing import Any, Optional
import asyncio
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import librosa
import soundfile as sf
import io
import base64
import tempfile
import os

from inference_types import (
    TTSResponse,
    OnlineTTSRequest,
    OfflineTTSRequest,
    TTSInferenceRequest,
    TTSProcessRequest,
    TTSPostProcessRequest,
)
# Note: These utilities should be imported from the main orchestrator
# For now, we'll use placeholder implementations that will be replaced
# by the actual F5TTSOrchestrator methods


@enum.unique
class TTSEngineMode(enum.Enum):
    OFFLINE = enum.auto()
    ONLINE = enum.auto()


@dataclasses.dataclass
class OfflineChannel:
    req_queue: queue.Queue[OfflineTTSRequest]
    res_queue: queue.Queue[TTSResponse]


@dataclasses.dataclass
class OnlineChannel:
    req_queue: asyncio.Queue[OnlineTTSRequest]
    aio_loop: asyncio.AbstractEventLoop


class TTSEngine:
    """TTS Engine for F5-TTS inference with streaming support."""

    def __init__(
        self,
        mesh: Mesh,
        orchestrator: Any,  # F5TTSOrchestrator instance
        mode: TTSEngineMode,
        channel: OfflineChannel | OnlineChannel,
        max_concurrent_requests: int = 4,
        chunk_duration: float = 1.0,  # Duration of each audio chunk in seconds
    ):
        print("Initializing TTS Engine")
        self.mesh = mesh
        self.orchestrator = orchestrator
        self.mode = mode
        self.channel = channel
        self.max_concurrent_requests = max_concurrent_requests
        self.chunk_duration = chunk_duration
        self.target_sr = 24000
        
        # Request management
        self.requests_dict: dict[str, TTSInferenceRequest] = {}
        self._max_requests_sem = threading.Semaphore(max_concurrent_requests)
        
        # Processing queues
        self._preprocess_queue: queue.Queue[TTSInferenceRequest] = queue.Queue()
        self._inference_queue: queue.Queue[TTSProcessRequest] = queue.Queue()
        self._postprocess_queue: queue.Queue[TTSPostProcessRequest] = queue.Queue(8)
        
        # Thread management
        self._setup_threads()
        
        # Statistics
        self.total_requests = 0
        self.completed_requests = 0
        self.start_time = None
        
        print("TTS Engine initialized")

    def _setup_threads(self):
        """Setup processing threads."""
        print("Setting up threads:", end="")
        
        if self.mode == TTSEngineMode.OFFLINE:
            print(" dequeue_offline,", end="")
            self._dequeue_thread = threading.Thread(
                name="dequeue_offline_request",
                target=self._dequeue_offline_request
            )
        else:
            print(" dequeue_online,", end="")
            self._dequeue_thread = threading.Thread(
                name="dequeue_online_request",
                target=self._dequeue_online_request
            )
        
        print(" preprocess,", end="")
        self._preprocess_thread = threading.Thread(
            name="preprocess",
            target=self._preprocess
        )
        
        print(" inference,", end="")
        self._inference_thread = threading.Thread(
            name="inference",
            target=self._inference
        )
        
        print(" postprocess")
        self._postprocess_thread = threading.Thread(
            name="postprocess",
            target=self._postprocess
        )

    def start(self):
        """Start the TTS engine."""
        print("Starting TTS Engine threads...")
        
        self._dequeue_thread.start()
        self._preprocess_thread.start()
        self._inference_thread.start()
        self._postprocess_thread.start()
        
        self.start_time = datetime.datetime.now()
        print(f"TTS Engine started: {self.start_time}")

    def stop(self):
        """Stop the TTS engine."""
        print("Stopping TTS Engine...")
        
        # Send stop signals
        if self.mode == TTSEngineMode.OFFLINE:
            self.channel.req_queue.put(None, block=True)
        else:
            self.channel.req_queue.put_nowait(None)
            
        self._preprocess_queue.put(None, block=True)
        self._inference_queue.put(None, block=True)
        self._postprocess_queue.put(None, block=True)
        
        # Wait for threads to finish
        self._dequeue_thread.join()
        self._preprocess_thread.join()
        self._inference_thread.join()
        self._postprocess_thread.join()
        
        stop_time = datetime.datetime.now()
        duration = (stop_time - self.start_time).total_seconds()
        print(f"TTS Engine stopped: {stop_time}")
        print(f"Total runtime: {duration:.2f} seconds")
        print(f"Processed {self.completed_requests}/{self.total_requests} requests")

    def _dequeue_online_request(self):
        """Dequeue online requests and put them in preprocess queue."""
        while True:
            try:
                r = asyncio.run_coroutine_threadsafe(
                    self.channel.req_queue.get(),
                    self.channel.aio_loop
                ).result()
                
                if r is None:
                    return
                    
                assert isinstance(r, OnlineTTSRequest)
                
                # Decode reference audio
                ref_audio, ref_sr = self._decode_audio_base64(r.ref_audio_base64)
                
                req = TTSInferenceRequest(
                    id=uuid.uuid4().hex,
                    ref_text=r.ref_text,
                    gen_text=r.gen_text,
                    ref_audio=ref_audio,
                    ref_sr=ref_sr,
                    num_inference_steps=r.num_inference_steps,
                    guidance_scale=r.guidance_scale,
                    speed_factor=r.speed_factor,
                    use_sway_sampling=r.use_sway_sampling,
                    aio_response_queue=r.res_queue,
                )
                
                self._preprocess_queue.put(req, block=True)
                self.requests_dict[req.id] = req
                self.total_requests += 1
                
            except Exception as e:
                print(f"Error in dequeue_online_request: {e}")
                break

    def _dequeue_offline_request(self):
        """Dequeue offline requests and put them in preprocess queue."""
        while True:
            r = self.channel.req_queue.get(block=True)
            if r is None:
                return
                
            assert isinstance(r, OfflineTTSRequest)
            
            # Decode reference audio
            ref_audio, ref_sr = self._decode_audio_base64(r.ref_audio_base64)
            
            req = TTSInferenceRequest(
                id=uuid.uuid4().hex,
                ref_text=r.ref_text,
                gen_text=r.gen_text,
                ref_audio=ref_audio,
                ref_sr=ref_sr,
                num_inference_steps=r.num_inference_steps,
                guidance_scale=r.guidance_scale,
                speed_factor=r.speed_factor,
                use_sway_sampling=r.use_sway_sampling,
            )
            
            self._preprocess_queue.put(req, block=True)
            self.requests_dict[req.id] = req
            self.total_requests += 1

    def _decode_audio_base64(self, audio_base64: str) -> tuple[np.ndarray, int]:
        """Decode base64 audio data."""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                
                audio, sr = librosa.load(tmp_file.name, sr=None)
                os.unlink(tmp_file.name)
                
                return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to decode audio: {e}")

    def _preprocess(self):
        """Preprocess requests for inference."""
        while True:
            req = self._preprocess_queue.get(block=True)
            if req is None:
                return
                
            try:
                self._max_requests_sem.acquire()
                
                # Resample reference audio if needed
                if req.ref_sr != self.target_sr:
                    ref_audio = librosa.resample(
                        req.ref_audio, orig_sr=req.ref_sr, target_sr=self.target_sr
                    )
                else:
                    ref_audio = req.ref_audio
                
                # Use orchestrator for mel spectrogram and text processing
                # This will be handled by the orchestrator's generate_audio method
                # For now, we'll pass the raw audio and text to the inference stage
                device_ref_audio = jax.device_put(
                    jnp.array(ref_audio), NamedSharding(self.mesh, P())
                )
                
                # Calculate chunk size for streaming
                chunk_size = int(self.chunk_duration * self.target_sr)
                
                process_req = TTSProcessRequest(
                    id=req.id,
                    ref_audio=ref_audio,
                    ref_sr=self.target_sr,
                    ref_text=req.ref_text,
                    gen_text=req.gen_text,
                    num_inference_steps=req.num_inference_steps,
                    guidance_scale=req.guidance_scale,
                    speed_factor=req.speed_factor,
                    use_sway_sampling=req.use_sway_sampling,
                    chunk_size=chunk_size,
                )
                
                self._inference_queue.put(process_req, block=True)
                
            except Exception as e:
                print(f"Error in preprocess: {e}")
                self._max_requests_sem.release()

    def _inference(self):
        """Run TTS inference."""
        while True:
            req = self._inference_queue.get(block=True)
            if req is None:
                return
                
            try:
                start_time = time.perf_counter()
                
                # Run F5-TTS inference using orchestrator
                sample_rate, generated_audio, generation_time = self.orchestrator.generate_audio(
                    ref_text=req.ref_text,
                    gen_text=req.gen_text,
                    ref_audio=np.array(req.ref_audio),
                    ref_sr=req.ref_sr,
                    num_inference_steps=req.num_inference_steps,
                    guidance_scale=req.guidance_scale,
                    speed_factor=req.speed_factor,
                    use_sway_sampling=req.use_sway_sampling,
                )
                
                # Split audio into chunks for streaming
                audio_chunks = []
                if len(generated_audio) > 0:
                    for i in range(0, len(generated_audio), req.chunk_size):
                        chunk = generated_audio[i:i + req.chunk_size]
                        audio_chunks.append(chunk)
                
                post_req = TTSPostProcessRequest(
                    request_id=req.id,
                    audio_chunks=audio_chunks,
                    sample_rate=sample_rate,
                    is_final=True,
                    generation_time=generation_time,
                )
                
                self._postprocess_queue.put(post_req, block=True)
                
            except Exception as e:
                print(f"Error in inference: {e}")
                # Send error response
                post_req = TTSPostProcessRequest(
                    request_id=req.id,
                    audio_chunks=[],
                    sample_rate=self.target_sr,
                    is_final=True,
                    generation_time=0.0,
                )
                self._postprocess_queue.put(post_req, block=True)

    def _postprocess(self):
        """Post-process inference results and send responses."""
        while True:
            post_req = self._postprocess_queue.get(block=True)
            if post_req is None:
                return
                
            try:
                req = self.requests_dict.get(post_req.request_id)
                if not req:
                    continue
                
                # Send audio chunks
                for i, chunk in enumerate(post_req.audio_chunks):
                    is_final_chunk = (i == len(post_req.audio_chunks) - 1) and post_req.is_final
                    
                    # Convert audio chunk to bytes
                    audio_bytes = self._audio_to_bytes(chunk, post_req.sample_rate)
                    
                    response = TTSResponse(
                        audio_chunk=audio_bytes,
                        sample_rate=post_req.sample_rate,
                        is_final=is_final_chunk,
                        metadata={
                            "chunk_index": i,
                            "total_chunks": len(post_req.audio_chunks),
                            "generation_time": post_req.generation_time,
                        } if is_final_chunk else {"chunk_index": i},
                    )
                    
                    if self.mode == TTSEngineMode.ONLINE and req.aio_response_queue:
                        self.channel.aio_loop.call_soon_threadsafe(
                            req.aio_response_queue.put_nowait, response
                        )
                    elif self.mode == TTSEngineMode.OFFLINE:
                        self.channel.res_queue.put_nowait(response)
                
                # Send final signal for online mode
                if self.mode == TTSEngineMode.ONLINE and req.aio_response_queue:
                    self.channel.aio_loop.call_soon_threadsafe(
                        req.aio_response_queue.put_nowait, None
                    )
                
                # Cleanup
                req.completed = True
                self.completed_requests += 1
                del self.requests_dict[post_req.request_id]
                self._max_requests_sem.release()
                
            except Exception as e:
                print(f"Error in postprocess: {e}")
                self._max_requests_sem.release()

    def _audio_to_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio array to bytes."""
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        return buffer.getvalue()