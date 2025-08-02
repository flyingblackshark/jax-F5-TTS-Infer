"""Example client for F5-TTS streaming API."""

import asyncio
import aiohttp
import json
import base64
import librosa
import soundfile as sf
import numpy as np
import io
import time
from typing import List


class F5TTSStreamingClient:
    """Client for F5-TTS streaming API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def encode_audio_file(self, file_path: str) -> str:
        """Encode audio file to base64."""
        audio, sr = librosa.load(file_path, sr=24000)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV')
        audio_bytes = buffer.getvalue()
        
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            async with self.session.get(f"{self.base_url}/v1/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("is_live") == "True"
                return False
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    async def generate_streaming(self, ref_text: str, gen_text: str, ref_audio_path: str,
                               num_inference_steps: int = 50, guidance_scale: float = 2.0,
                               speed_factor: float = 1.0, use_sway_sampling: bool = False) -> List[bytes]:
        """Generate TTS audio with streaming."""
        # Encode reference audio
        ref_audio_base64 = self.encode_audio_file(ref_audio_path)
        
        # Prepare request data
        request_data = {
            "ref_text": ref_text,
            "gen_text": gen_text,
            "ref_audio_base64": ref_audio_base64,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "speed_factor": speed_factor,
            "use_sway_sampling": use_sway_sampling
        }
        
        audio_chunks = []
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/generate/stream",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Server error {response.status}: {error_text}")
                
                print("Receiving streaming audio chunks...")
                chunk_count = 0
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        try:
                            chunk_data = json.loads(data_str)
                            
                            if 'audio_chunk' in chunk_data:
                                # Decode base64 audio chunk
                                audio_chunk = base64.b64decode(chunk_data['audio_chunk'])
                                audio_chunks.append(audio_chunk)
                                chunk_count += 1
                                
                                print(f"Received chunk {chunk_count} (size: {len(audio_chunk)} bytes)")
                                
                                if chunk_data.get('is_final', False):
                                    metadata = chunk_data.get('metadata', {})
                                    generation_time = metadata.get('generation_time', 0)
                                    total_chunks = metadata.get('total_chunks', chunk_count)
                                    
                                    print(f"Final chunk received. Total chunks: {total_chunks}")
                                    print(f"Generation time: {generation_time:.2f}s")
                                    break
                        
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse chunk data: {e}")
                            continue
        
        except Exception as e:
            print(f"Streaming error: {e}")
            raise
        
        total_time = time.time() - start_time
        print(f"Total streaming time: {total_time:.2f}s")
        print(f"Received {len(audio_chunks)} audio chunks")
        
        return audio_chunks
    
    async def generate_non_streaming(self, ref_text: str, gen_text: str, ref_audio_path: str,
                                   num_inference_steps: int = 50, guidance_scale: float = 2.0,
                                   speed_factor: float = 1.0, use_sway_sampling: bool = False) -> dict:
        """Generate TTS audio without streaming."""
        # Encode reference audio
        ref_audio_base64 = self.encode_audio_file(ref_audio_path)
        
        # Prepare request data
        request_data = {
            "ref_text": ref_text,
            "gen_text": gen_text,
            "ref_audio_base64": ref_audio_base64,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "speed_factor": speed_factor,
            "use_sway_sampling": use_sway_sampling
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Server error {response.status}: {error_text}")
                
                result = await response.json()
                
                total_time = time.time() - start_time
                print(f"Non-streaming generation completed in {total_time:.2f}s")
                print(f"Server generation time: {result.get('generation_time', 0):.2f}s")
                print(f"Audio duration: {result.get('duration', 0):.2f}s")
                print(f"Sample rate: {result.get('sample_rate', 0)} Hz")
                
                return result
        
        except Exception as e:
            print(f"Generation error: {e}")
            raise
    
    def save_audio_chunks(self, audio_chunks: List[bytes], output_path: str, sample_rate: int = 24000):
        """Save audio chunks to a file."""
        if not audio_chunks:
            print("No audio chunks to save")
            return
        
        # Combine all chunks
        combined_audio = b''.join(audio_chunks)
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(combined_audio)
        
        print(f"Saved combined audio to {output_path} ({len(combined_audio)} bytes)")
    
    def save_audio_base64(self, audio_base64: str, output_path: str):
        """Save base64 audio to a file."""
        audio_bytes = base64.b64decode(audio_base64)
        
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"Saved audio to {output_path} ({len(audio_bytes)} bytes)")


async def main():
    """Example usage of the streaming client."""
    # Configuration
    ref_text = "Hello, this is a reference text for voice cloning."
    gen_text = "This is the text that will be generated with the cloned voice."
    ref_audio_path = "reference_audio.wav"  # Path to your reference audio file
    
    async with F5TTSStreamingClient() as client:
        # Health check
        print("Checking server health...")
        if not await client.health_check():
            print("Server is not healthy. Please start the server first.")
            return
        
        print("Server is healthy!")
        
        try:
            # Test streaming generation
            print("\n=== Testing Streaming Generation ===")
            audio_chunks = await client.generate_streaming(
                ref_text=ref_text,
                gen_text=gen_text,
                ref_audio_path=ref_audio_path,
                num_inference_steps=32,
                guidance_scale=2.0,
                speed_factor=1.0
            )
            
            # Save streaming result
            client.save_audio_chunks(audio_chunks, "output_streaming.wav")
            
            # Test non-streaming generation
            print("\n=== Testing Non-Streaming Generation ===")
            result = await client.generate_non_streaming(
                ref_text=ref_text,
                gen_text=gen_text,
                ref_audio_path=ref_audio_path,
                num_inference_steps=32,
                guidance_scale=2.0,
                speed_factor=1.0
            )
            
            # Save non-streaming result
            client.save_audio_base64(result['audio_base64'], "output_non_streaming.wav")
            
        except Exception as e:
            print(f"Error during generation: {e}")


if __name__ == "__main__":
    # Note: Make sure you have a reference audio file named 'reference_audio.wav'
    # in the same directory, or update the ref_audio_path variable
    asyncio.run(main())