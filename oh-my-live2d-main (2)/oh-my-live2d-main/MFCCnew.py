import numpy as np
import librosa
import pandas as pd
import os

# 可选：如果你想把重采样后的音频也保存出来
SAVE_RESAMPLED_WAV = False

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    SAVE_RESAMPLED_WAV = False  # 没装 soundfile 就自动不保存重采样音频


def extract_mfcc_features(file_path, max_pad_len=128, target_sr=44100, n_mfcc=128):
    """
    读取音频 -> 统一重采样到 target_sr -> 提取 MFCC
    返回:
        mfccs, original_sr, target_sr
    """
    try:
        # 先用原采样率读一次，获取 original_sr（用于输出）
        audio_raw, original_sr = librosa.load(file_path, sr=None, mono=True)

        # 再重采样到 44100
        if original_sr != target_sr:
            audio = librosa.resample(audio_raw, orig_sr=original_sr, target_sr=target_sr)
        else:
            audio = audio_raw

        # 计算 MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc)

        # pad / truncate 到固定帧数
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.astype(np.float32), original_sr, target_sr, audio

    except Exception as e:
        print(f"❌ Error while parsing: {file_path}\nException: {e}")
        return None, None, None, None


# ============ 路径设置 ============
directory = r"D:\比赛视频Videos\sopran_cutted"
Audio_path = os.path.join(directory, "Audio")

MFCC_out = os.path.join(directory, "MFCC_Output")
os.makedirs(MFCC_out, exist_ok=True)

Resampled_out = os.path.join(directory, "Audio_44100")
if SAVE_RESAMPLED_WAV:
    os.makedirs(Resampled_out, exist_ok=True)

files = os.listdir(Audio_path)
wav_files = [f for f in files if f.lower().endswith(".wav")]

print(f"📁 输入目录: {Audio_path}")
print(f"📦 共检测到 WAV 文件: {len(wav_files)}")
print(f"🧾 MFCC 输出目录: {MFCC_out}")
if SAVE_RESAMPLED_WAV:
    print(f"🎧 重采样音频输出目录: {Resampled_out} (44100Hz)")
print("-" * 60)

# ============ 主循环 ============
for idx, file in enumerate(wav_files, start=1):
    in_path = os.path.join(Audio_path, file)

    mfccs, original_sr, target_sr, audio_44100 = extract_mfcc_features(
        in_path,
        max_pad_len=128,
        target_sr=44100,
        n_mfcc=128
    )

    if mfccs is None:
        continue

    # 保存 MFCC Excel
    out_name = os.path.splitext(file)[0] + "_MFCC.xlsx"
    out_path = os.path.join(MFCC_out, out_name)
    pd.DataFrame(mfccs).to_excel(out_path, index=False, header=False)

    # 可选：保存重采样后的 wav
    if SAVE_RESAMPLED_WAV and HAS_SF:
        wav_out_path = os.path.join(Resampled_out, file)
        # 用 PCM_16 更通用
        sf.write(wav_out_path, audio_44100, target_sr, subtype="PCM_16")

    # 增加输出信息
    print(
        f"✅ [{idx}/{len(wav_files)}] {file}\n"
        f"   原采样率: {original_sr} Hz -> 重采样: {target_sr} Hz\n"
        f"   MFCC shape: {mfccs.shape} (n_mfcc, frames)\n"
        f"   MFCC saved: {out_path}"
        + (f"\n   WAV saved:  {wav_out_path}" if SAVE_RESAMPLED_WAV and HAS_SF else "")
        + "\n"
    )

print("🎉 全部处理完成。")
