#!/usr/bin/env python3
"""Example client for F5-TTS FastAPI service"""

import requests
import base64
import io
import soundfile as sf
import argparse
import os

def generate_tts_form_data(server_url: str, ref_text: str, gen_text: str, ref_audio_path: str, 
                          num_inference_steps: int = 50, guidance_scale: float = 2.0,
                          speed_factor: float = 1.0, use_sway_sampling: bool = False,
                          output_path: str = "output.wav"):
    """Generate TTS using form data endpoint"""
    
    url = f"{server_url}/generate"
    
    # Read and encode audio file to base64
    with open(ref_audio_path, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare form data
    data = {
        'ref_text': ref_text,
        'gen_text': gen_text,
        'ref_audio_base64': audio_base64,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'speed_factor': speed_factor,
        'use_sway_sampling': use_sway_sampling
    }
    
    print(f"Sending request to {url}...")
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        result = response.json()
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(result['audio_base64'])
        
        # Save audio file
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"✅ Audio generated successfully!")
        print(f"📁 Saved to: {output_path}")
        print(f"⏱️  Generation time: {result['generation_time']:.2f}s")
        print(f"🎵 Audio duration: {result['duration']:.2f}s")
        print(f"🔊 Sample rate: {result['sample_rate']} Hz")
        
        return True
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def generate_tts_json(server_url: str, ref_text: str, gen_text: str, ref_audio_path: str,
                     num_inference_steps: int = 50, guidance_scale: float = 2.0,
                     speed_factor: float = 1.0, use_sway_sampling: bool = False,
                     output_path: str = "output.wav"):
    """Generate TTS using JSON endpoint"""
    
    url = f"{server_url}/generate_json"
    
    # Read and encode audio file to base64
    with open(ref_audio_path, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare JSON data
    json_data = {
        'ref_text': ref_text,
        'gen_text': gen_text,
        'ref_audio_base64': audio_base64,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'speed_factor': speed_factor,
        'use_sway_sampling': use_sway_sampling
    }
    
    print(f"Sending request to {url}...")
    response = requests.post(
        url, 
        json=json_data, 
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(result['audio_base64'])
        
        # Save audio file
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"✅ Audio generated successfully!")
        print(f"📁 Saved to: {output_path}")
        print(f"⏱️  Generation time: {result['generation_time']:.2f}s")
        print(f"🎵 Audio duration: {result['duration']:.2f}s")
        print(f"🔊 Sample rate: {result['sample_rate']} Hz")
        
        return True
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def check_server_health(server_url: str):
    """Check if the server is running and healthy"""
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Server is healthy")
            print(f"📊 Status: {result['status']}")
            print(f"🤖 Model loaded: {result['model_loaded']}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {server_url}")
        return False
    except Exception as e:
        print(f"❌ Error checking server health: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="F5-TTS FastAPI Client Example")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--ref-text", required=True, help="Reference text")
    parser.add_argument("--gen-text", required=True, help="Text to generate")
    parser.add_argument("--ref-audio", required=True, help="Reference audio file path")
    parser.add_argument("--output", default="output.wav", help="Output audio file path")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="Guidance scale")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor")
    parser.add_argument("--sway", action="store_true", help="Use sway sampling")
    parser.add_argument("--method", choices=["form", "json"], default="form", help="Request method")
    parser.add_argument("--check-health", action="store_true", help="Only check server health")
    
    args = parser.parse_args()
    
    # Check server health first
    if not check_server_health(args.server):
        return
    
    if args.check_health:
        return
    
    # Validate input files
    if not os.path.exists(args.ref_audio):
        print(f"❌ Reference audio file not found: {args.ref_audio}")
        return
    
    print(f"🎯 Reference text: {args.ref_text}")
    print(f"📝 Generation text: {args.gen_text}")
    print(f"🎵 Reference audio: {args.ref_audio}")
    print(f"⚙️  Steps: {args.steps}, CFG: {args.cfg}, Speed: {args.speed}, Sway: {args.sway}")
    print(f"🔧 Method: {args.method}")
    print()
    
    # Generate TTS
    if args.method == "form":
        success = generate_tts_form_data(
            server_url=args.server,
            ref_text=args.ref_text,
            gen_text=args.gen_text,
            ref_audio_path=args.ref_audio,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            speed_factor=args.speed,
            use_sway_sampling=args.sway,
            output_path=args.output
        )
    else:  # json
        success = generate_tts_json(
            server_url=args.server,
            ref_text=args.ref_text,
            gen_text=args.gen_text,
            ref_audio_path=args.ref_audio,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            speed_factor=args.speed,
            use_sway_sampling=args.sway,
            output_path=args.output
        )
    
    if success:
        print(f"\n🎉 TTS generation completed successfully!")
    else:
        print(f"\n💥 TTS generation failed!")

if __name__ == "__main__":
    main()