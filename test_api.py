#!/usr/bin/env python3
"""Test script for F5-TTS FastAPI service"""

import requests
import base64
import io
import numpy as np
import soundfile as sf
import tempfile
import os
import time
import json
from typing import Optional

def create_test_audio(duration: float = 2.0, sample_rate: int = 24000, frequency: float = 440.0) -> bytes:
    """Create a test audio file (sine wave)"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Convert to bytes
    with io.BytesIO() as buffer:
        sf.write(buffer, audio, sample_rate, format='WAV')
        return buffer.getvalue()

def test_health_check(server_url: str) -> bool:
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {result.get('status')}")
            print(f"   Model loaded: {result.get('model_loaded')}")
            print(f"   Timestamp: {result.get('timestamp')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_form_data_generation(server_url: str, save_output: bool = True) -> bool:
    """Test audio generation via form data"""
    print("\n🎵 Testing form data generation...")
    
    try:
        # Create test audio
        test_audio_bytes = create_test_audio(duration=1.0)
        
        # Convert audio to base64
        audio_base64 = base64.b64encode(test_audio_bytes).decode('utf-8')
        
        # Prepare form data
        data = {
            'ref_text': '这是一个测试音频。',
            'gen_text': '你好，这是F5-TTS生成的语音。',
            'ref_audio_base64': audio_base64,
            'num_inference_steps': 32,  # Reduced for faster testing
            'guidance_scale': 2.0,
            'speed_factor': 1.0,
            'use_sway_sampling': False
        }
        
        print(f"   Sending request to {server_url}/generate...")
        start_time = time.time()
        
        response = requests.post(f"{server_url}/generate", data=data, timeout=120)
        
        end_time = time.time()
        request_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Form data generation successful")
            print(f"   Request time: {request_time:.2f}s")
            print(f"   Generation time: {result.get('generation_time', 'N/A')}s")
            print(f"   Audio duration: {result.get('duration', 'N/A')}s")
            print(f"   Sample rate: {result.get('sample_rate', 'N/A')} Hz")
            print(f"   Text length: {result.get('text_length', 'N/A')}")
            
            if save_output and 'audio_base64' in result:
                # Save generated audio
                audio_bytes = base64.b64decode(result['audio_base64'])
                output_path = 'test_output_form.wav'
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                print(f"   Saved audio to: {output_path}")
            
            return True
        else:
            print(f"❌ Form data generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Form data generation error: {e}")
        return False

def test_json_generation(server_url: str, save_output: bool = True) -> bool:
    """Test audio generation via JSON"""
    print("\n📝 Testing JSON generation...")
    
    try:
        # Create test audio
        test_audio_bytes = create_test_audio(duration=1.0)
        audio_base64 = base64.b64encode(test_audio_bytes).decode('utf-8')
        
        # Prepare JSON data
        json_data = {
            'ref_text': '这是一个测试音频。',
            'gen_text': '你好，这是通过JSON接口生成的语音。',
            'ref_audio_base64': audio_base64,
            'num_inference_steps': 32,  # Reduced for faster testing
            'guidance_scale': 2.0,
            'speed_factor': 1.0,
            'use_sway_sampling': False
        }
        
        print(f"   Sending request to {server_url}/generate_json...")
        start_time = time.time()
        
        response = requests.post(
            f"{server_url}/generate_json", 
            json=json_data, 
            headers={'Content-Type': 'application/json'},
            timeout=120
        )
        
        end_time = time.time()
        request_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ JSON generation successful")
            print(f"   Request time: {request_time:.2f}s")
            print(f"   Generation time: {result.get('generation_time', 'N/A')}s")
            print(f"   Audio duration: {result.get('duration', 'N/A')}s")
            print(f"   Sample rate: {result.get('sample_rate', 'N/A')} Hz")
            print(f"   Text length: {result.get('text_length', 'N/A')}")
            
            if save_output and 'audio_base64' in result:
                # Save generated audio
                audio_bytes = base64.b64decode(result['audio_base64'])
                output_path = 'test_output_json.wav'
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                print(f"   Saved audio to: {output_path}")
            
            return True
        else:
            print(f"❌ JSON generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ JSON generation error: {e}")
        return False

def test_error_handling(server_url: str) -> bool:
    """Test error handling"""
    print("\n🚨 Testing error handling...")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Missing required fields
    try:
        response = requests.post(f"{server_url}/generate", data={}, timeout=10)
        if response.status_code == 422:  # Validation error
            print("✅ Missing fields validation works")
            tests_passed += 1
        else:
            print(f"❌ Missing fields validation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Missing fields test error: {e}")
    
    # Test 2: Invalid audio format
    try:
        data = {
            'ref_text': '测试',
            'gen_text': '测试生成',
        }
        files = {
            'ref_audio': ('test.txt', b'invalid audio data', 'text/plain')
        }
        response = requests.post(f"{server_url}/generate", data=data, files=files, timeout=10)
        if response.status_code in [400, 422]:  # Bad request or validation error
            print("✅ Invalid audio format validation works")
            tests_passed += 1
        else:
            print(f"❌ Invalid audio format validation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid audio format test error: {e}")
    
    # Test 3: Invalid JSON
    try:
        response = requests.post(
            f"{server_url}/generate_json", 
            data="invalid json", 
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        if response.status_code == 422:  # Validation error
            print("✅ Invalid JSON validation works")
            tests_passed += 1
        else:
            print(f"❌ Invalid JSON validation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid JSON test error: {e}")
    
    print(f"   Error handling tests passed: {tests_passed}/{total_tests}")
    return tests_passed == total_tests

def test_performance(server_url: str, num_requests: int = 3) -> bool:
    """Test performance with multiple requests"""
    print(f"\n⚡ Testing performance ({num_requests} requests)...")
    
    try:
        test_audio_bytes = create_test_audio(duration=0.5)  # Short audio for speed
        audio_base64 = base64.b64encode(test_audio_bytes).decode('utf-8')
        
        times = []
        success_count = 0
        
        for i in range(num_requests):
            data = {
                'ref_text': f'测试音频{i+1}',
                'gen_text': f'这是第{i+1}个测试请求。',
                'ref_audio_base64': audio_base64,
                'num_inference_steps': 20,  # Very fast for testing
                'guidance_scale': 2.0,
                'speed_factor': 1.0,
                'use_sway_sampling': False
            }
            
            start_time = time.time()
            response = requests.post(f"{server_url}/generate", data=data, timeout=60)
            end_time = time.time()
            
            request_time = end_time - start_time
            times.append(request_time)
            
            if response.status_code == 200:
                success_count += 1
                print(f"   Request {i+1}: {request_time:.2f}s ✅")
            else:
                print(f"   Request {i+1}: {request_time:.2f}s ❌ ({response.status_code})")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"✅ Performance test completed")
            print(f"   Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
            print(f"   Average time: {avg_time:.2f}s")
            print(f"   Min time: {min_time:.2f}s")
            print(f"   Max time: {max_time:.2f}s")
            
            return success_count == num_requests
        else:
            print("❌ No requests completed")
            return False
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def main():
    """Run all tests"""
    server_url = "http://localhost:8000"
    
    print("🧪 F5-TTS FastAPI Service Test Suite")
    print("=" * 50)
    print(f"Server URL: {server_url}")
    print()
    
    # Test results
    results = {
        'health_check': False,
        'form_data_generation': False,
        'json_generation': False,
        'error_handling': False,
        'performance': False
    }
    
    # Run tests
    results['health_check'] = test_health_check(server_url)
    
    if results['health_check']:
        results['form_data_generation'] = test_form_data_generation(server_url)
        results['json_generation'] = test_json_generation(server_url)
        results['error_handling'] = test_error_handling(server_url)
        results['performance'] = test_performance(server_url, num_requests=2)
    else:
        print("\n❌ Skipping other tests due to health check failure")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print()
    
    passed_tests = 0
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
        if passed:
            passed_tests += 1
    
    print()
    print(f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The service is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the service configuration.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)