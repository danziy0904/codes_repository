import os
import librosa
import soundfile
import numpy as np

def creat_folder(fd):#创建文件
    if not os.path.exists(fd):
        os.makedirs(fd)

def read_audio(path,target_fs = None):
    (audio,fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio,axis = 1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample (audio,orig_sr=fs,target_sr=target_fs)
        fs = target_fs
    return audio ,fs#fs音频文件的采样率
