
import os
import sys
import wave
import json
import dashscope
from dashscope.audio.asr import Recognition
try:
    from dashscope.audio.asr import Transcription
except ImportError:
    Transcription = None

# ... (rest of imports)

def test_transcription(model_name, wav_path):
    if not Transcription:
        print("\nSkipping Transcription test (not imported)")
        return
        
    print(f"\nTesting Transcription model: {model_name}")
    try:
        # Transcription usually is for long file, async. But maybe 'flash' supports it?
        # Usually: Transcription.call(model=..., file_urls=[...]) or file_paths
        # But let's check doc/signature first.
        print("Calling Transcription.call(model=..., file_paths=[...])...")
        # Note: Transcription often returns a task_id, not immediate result.
        # But 'flash' models might be different.
        
        # We need a public URL for Transcription usually, but let's try local file if SDK supports it.
        # SDK might upload it automatically.
        import inspect
        print(f"Transcription.call signature: {inspect.signature(Transcription.call)}")
        
        # Assume it's class method
        resp = Transcription.call(model=model_name, file_paths=[wav_path])
        print_response(resp)
        
    except Exception as e:
        print(f"FAILED Transcription: {e}")

if __name__ == "__main__":
    import dashscope.audio.asr
    print(f"dashscope.audio.asr members: {dir(dashscope.audio.asr)}")
    
    wav_file = create_test_wav()
    print(f"Created temporary wav: {wav_file}")
    
    # Test qwen3-asr-flash with Recognition (Instance) - we know this fails with 44/ModelNotFound
    # test_model("qwen3-asr-flash", wav_file) 
    
    # Test Transcription
    test_transcription("qwen3-asr-flash", wav_file)
    test_transcription("qwen3-asr-flash-filetrans", wav_file)
    
    try:
        os.remove(wav_file)
    except:
        pass
