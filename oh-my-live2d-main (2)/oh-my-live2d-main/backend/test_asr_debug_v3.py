
import os
import sys
import wave
import json
import time
import dashscope
from dashscope.audio.asr import Recognition
try:
    from dashscope.audio.asr import Transcription
except ImportError:
    Transcription = None

# Load .env manually
try:
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
except Exception:
    pass

API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    print("Error: DASHSCOPE_API_KEY not found in .env or environment")
    sys.exit(1)

dashscope.api_key = API_KEY

# Ensure China Endpoint
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1" 

def create_test_wav(filename="test_audio.wav"):
    sample_rate = 16000
    duration = 2
    # Generate silence
    data = b'\x00' * (sample_rate * duration * 2)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    return filename

def print_response(resp):
    print(f"Response Type: {type(resp)}")
    if hasattr(resp, 'status_code'):
        print(f"Status Code: {resp.status_code}")
    if hasattr(resp, 'code'):
        print(f"Code: {resp.code}")
    if hasattr(resp, 'message'):
        print(f"Message: {resp.message}")
    if hasattr(resp, 'output'):
        print(f"Output: {resp.output}")

def test_transcription(model_name, wav_path):
    if not Transcription:
        print(f"\nSkipping Transcription test for {model_name} (not imported)")
        return
        
    print(f"\nTesting Transcription model: {model_name}")
    try:
        # Transcription.call usually takes file_urls or file_paths
        # Try local file path first
        print(f"Calling Transcription.call(model={model_name}, file_paths=[{wav_path}])...")
        resp = Transcription.call(model=model_name, file_paths=[wav_path])
        print_response(resp)
        
    except Exception as e:
        print(f"FAILED Transcription: {e}")

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # print(f"dashscope version: {dashscope.__version__}")
    print(f"Base URL: {dashscope.base_http_api_url}")
    
    print("Recognition Docstring:")
    print(Recognition.__doc__)
    
    wav_file = create_test_wav()
    
    # Test paraformer-realtime-v1 with Recognition (Instance)
    print("\nTesting paraformer-realtime-v1 with Recognition...")
    try:
        recog = Recognition(model='paraformer-realtime-v1', format='wav', sample_rate=16000, callback=None)
        resp = recog.call(file=wav_file)
        print_response(resp)
    except Exception as e:
        print(f"FAILED paraformer-realtime-v1: {e}")

    try:
        os.remove(wav_file)
    except:
        pass
