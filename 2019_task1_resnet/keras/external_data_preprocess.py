import librosa
import os
import soundfile


def split_external_data():
    path = '/home/r506/DCase/scenes_stereo/'
    files = os.listdir(path)
    sorted(files)
    for file in files[:1]:
        wav_path = os.path.join(path, file)
        y, sr = soundfile.read(wav_path, )
        # audio2 = librosa.load(wav_path, sr=44100, offset=10.0, mono=False, duration=10.0)
        # audio3 = librosa.load(wav_path, sr=44100, offset=20.0, mono=False, duration=10.0)
        path1 = os.path.join(path, file.split('.')[0] + '_0.wav')
        # path2 = os.path.join(path, file.split('.')[0] + "_10.wav")
        # path3 = os.path.join(path, file.split('.')[0] + "_20")
        # librosa.output.write_wav(path1, y=audio1[0], sr=44100)
        # librosa.output.write_wav(path2, y=audio2[0], sr=44100)
        # librosa.output.write_wav(path3, y=audio3[0], sr=44100)
        # print(audio1)
        soundfile.write(path1, y, samplerate=sr)


if __name__ == '__main__':
    # split_external_data()
    path = '/home/r506/DCase/scenes_stereo/'
    data, sr = librosa.load(os.path.join(path, 'tubestation07.wav'), mono=False, offset=10.0, sr=44100, duration=10.0)
    print(data.shape)
    soundfile.write('tubestation.wav', data, samplerate=sr)
    # print(data.shape)
