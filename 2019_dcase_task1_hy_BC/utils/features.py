import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import time
import csv
# from torch import nn
# from torch.autograd import Variable
# import torch as t
# import pywt

from utilities import read_audio, create_folder
import config


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        self.melW = librosa.filters.mel(sr=sample_rate,
                                        n_fft=window_size,
                                        n_mels=mel_bins,
                                        fmin=50.,
                                        fmax=sample_rate // 2).T

    def transform(self, audio):
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap

        [f, t, x] = signal.spectral.spectrogram(
            audio,
            window=ham_win,
            nperseg=window_size,
            noverlap=overlap,
            detrend=False,
            return_onesided=True,
            mode='magnitude')
        x = x.T

        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)

        return x


def calculate_hpss_logmel(audio_path, sample_rate, feature_extractor):
    # Read stereo audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''

    # Extract feature
    h, p = librosa.effects.hpss(audio)
    h_feature = feature_extractor.transform(h)
    p_feature = feature_extractor.transform(p)

    return np.stack([h_feature, p_feature], axis=0)


def calculate_three_logmel(audio_path, sample_rate, feature_extractor):
    # Read stereo audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''
    # Extract feature
    l_r = audio[0] + audio[1]
    audio = np.mean(audio, axis=0)
    h, p = librosa.effects.hpss(audio)
    h_feature = feature_extractor.transform(h)
    p_feature = feature_extractor.transform(p)
    l_r_feature = feature_extractor.transform(l_r)
    # mono_feature = feature_extractor.transform(audio)

    return np.stack([h_feature, p_feature, l_r_feature], axis=0)


def calculate_lr_logmel(audio_path, sample_rate, feature_extractor):
    # Read stereo audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''
    # Extract feature
    l_feature = feature_extractor.transform(audio[0])
    r_feature = feature_extractor.transform(audio[1])
    # l_r_feature = feature_extractor.transform(l_r)
    # mono_feature = feature_extractor.transform(audio)

    return np.stack([l_feature, r_feature], axis=0)
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight

def calculate_three_logmel_BC(audio_path, sample_rate, feature_extractor):
    # Read stereo audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)
    #添加A加权代码
    fs = 44100
    n_fft = 2048  # 论文参数是4096
    min_db = -80.0
    stride = n_fft // 2
    gain = []
    for i in range(0, len(audio) - n_fft + 1, stride):
        spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
        # print(spec.shape)  # (1025)
        power_spec = np.abs(spec) ** 2
        a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
        # print(30, a_weighted_spec.shape)  # (1025)
        g = np.sum(a_weighted_spec)
        #print(i, g)

        gain.append(g)
    # print(i)
    gain = np.array(gain)
    # print(gain.shape)  # (214,)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)
    # print(gain_db)
    gain_db = np.max(gain_db)
    # print(42, gain)
    # print(gain.shape)

    # gain_db得出来得频域信息

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''
    # Extract feature
    l_r = audio[0] + audio[1]
    audio = np.mean(audio, axis=0)
    h, p = librosa.effects.hpss(audio)
    h_feature = feature_extractor.transform(h)
    p_feature = feature_extractor.transform(p)
    l_r_feature = feature_extractor.transform(l_r)

    return gain_db,np.stack([h_feature, p_feature, l_r_feature], axis=0)


# class NetG(nn.Module):
#     def __init__(self):
#         super(NetG, self).__init__()
#         self.ConTh = nn.Sequential(
#             nn.ConvTranspose2d(3, 3, 2, 2, 2, bias=False),
#             nn.BatchNorm2d(3),
#             nn.ReLU(True)
#         )
#
#     def forward(self, input):
#         return self.ConTh(input)


# def calculate_logmel(audio_path,sample_rate,feature_extractor):
#     (audio, fs) = read_audio(audio_path, target_fs=sample_rate)
#
#    # print(audio.shape)#(2,441000)
#     audio = np.mean(audio,axis = 0)
#     # print(audio)
#     # print(audio.shape)
#     feature = feature_extractor.transform(audio)
#     #print(160, feature.shape)
#     # input_layer = Input(shape=(3, 162,34))
#     # feature = keras.layers.convolutional.SeparableConv2D(input= input_layer,filters=3, kernel_size=2, strides=(2, 2), input_shape=(3,162,34),padding=(2,2), data_format="channel_first",
#     #                                            depth_multiplier=1, activation=None, use_bias=True,
#     #                                            depthwise_initializer='glorot_uniform',
#     #                                            pointwise_initializer='glorot_uniform', bias_initializer='zeros',
#     #                                            depthwise_regularizer=None, pointwise_regularizer=None,
#     #                                            bias_regularizer=None, activity_regularizer=None,
#     #                                            depthwise_constraint=None, pointwise_constraint=None,
#     #                                            bias_constraint=None)
#     # feature = keras.layers.convolutional.Conv2DTranspose(input_shape=(3,162,34),filters=3, kernel_size=2,strides=(2, 2), padding="same", data_format=None )(feature)
#     # print(141,feature.shape)
#
#     # titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
#     coeffs2 = pywt.dwt2(feature, 'bior1.3')
#     LL, (LH, HL, HH) = coeffs2  # 分别是水平，竖直，对角
#     feature = np.stack([LH, HL, HH], axis=0)  # pytorch中通道是最后一维
#     feature = t.from_numpy(feature)
#     feature = feature.unsqueeze(0)
#     net = NetG()
#     feature = net(feature)
#     feature = feature.squeeze(0)
#     feature = feature.detach().numpy()
#     print(feature.shape)
#
#     # print(LL)
#     # print(LL.shape)  # (34,162)
#     # fig = plt.figure(figsize=(12, 3))
#     # #for i, a in enumerate([LL, LH, HL, HH]):
#     # for i, a in enumerate(feature_array):
#     #     ax = fig.add_subplot(1, 4, i + 1)
#     #     ax.imshow(a, interpolation="nearest")
#     #     ax.set_title(titles[i], fontsize=10)
#     #     ax.set_xticks([])
#     #     ax.set_yticks([])
#     # fig.tight_layout()
#     # plt.show()
#     # feature = np.stack([LH, HL,HH], axis=0)
#     # print(feature.shape)
#
#     return feature


def read_development_meta(meta_csv):
    df = pd.read_csv(meta_csv, sep='\t')
    df = pd.DataFrame(df)

    audio_names = []
    scene_labels = []
    identifiers = []
    source_labels = []

    for row in df.iterrows():
        audio_name = row[1]['filename'].split('/')[1]
        scene_label = row[1]['scene_label']
        identifier = row[1]['identifier']
        source_label = row[1]['source_label']

        audio_names.append(audio_name)
        scene_labels.append(scene_label)
        identifiers.append(identifier)
        source_labels.append(source_label)

    return audio_names, scene_labels, identifiers, source_labels


def read_evaluation_meta(evaluation_csv):
    with open(evaluation_csv, 'r') as f:
        reader = csv.reader(f)
        lis = list(reader)

    audio_names = []
    lis.pop(0)
    for li in lis:
        audio_name = li[0].split('/')[1]
        audio_names.append(audio_name)

    return audio_names


def calculate_multi_features(args):
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins

    # Paths
    audio_dir = os.path.join(dataset_dir, subdir, 'audio')

    if data_type == 'development':
        meta_csv = os.path.join(dataset_dir, subdir, 'meta.csv')

    elif data_type in ['leaderboard', 'evaluation']:
        evaluation_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                      'test.csv')

    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'mini_BClearning_{}.h5'.format(data_type))
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 '{}_dct_2019.h5'.format(data_type))

    create_folder(os.path.dirname(hdf5_path))

    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate,
                                        window_size=window_size,
                                        overlap=overlap,
                                        mel_bins=mel_bins)

    # Read meta csv
    if data_type == 'development':
        [audio_names, scene_labels, identifiers, source_labels] = \
            read_development_meta(meta_csv)

    elif data_type in ['leaderboard', 'evaluation']:
        audio_names = read_evaluation_meta(evaluation_csv)

    # Only use partial data when set mini_data to True
    if mini_data:

        audios_num = 300
        random_state = np.random.RandomState(0)
        audio_indexes = np.arange(len(audio_names))
        random_state.shuffle(audio_indexes)
        audio_indexes = audio_indexes[0: audios_num]

        audio_names = [audio_names[idx] for idx in audio_indexes]

        if data_type == 'development':
            scene_labels = [scene_labels[idx] for idx in audio_indexes]
            identifiers = [identifiers[idx] for idx in audio_indexes]
            source_labels = [source_labels[idx] for idx in audio_indexes]

    print('Number of audios: {}'.format(len(audio_names)))

    # Create hdf5 file
    hf = h5py.File(hdf5_path, 'w')

    hf.create_dataset(
        name='feature',
        shape=(0, 3, seq_len, mel_bins),
        maxshape=(None, 3, seq_len, mel_bins),
        dtype=np.float32)

    calculate_time = time.time()

    for (n, audio_name) in enumerate(audio_names):
        print(n, audio_name)

        # Calculate feature
        audio_path = os.path.join(audio_dir, audio_name)

        # Extract feature
        feature = calculate_three_logmel(audio_path=audio_path,
                                        sample_rate=sample_rate,
                                         feature_extractor=feature_extractor)
        '''(seq_len, mel_bins)'''
        print("calculate_three_logmel")
        print(feature.shape)

        hf['feature'].resize((n + 1, 3, seq_len, mel_bins))
        hf['feature'][n] = feature

    # Write meta info to hdf5
    hf.create_dataset(name='filename',
                      data=[s.encode() for s in audio_names],
                      dtype='S50')
    if data_type == 'development':
        hf.create_dataset(name='scene_label',
                          data=[s.encode() for s in scene_labels],
                          dtype='S20')

        hf.create_dataset(name='identifier',
                          data=[s.encode() for s in identifiers],
                          dtype='S20')

        hf.create_dataset(name='source_label',
                          data=[s.encode() for s in source_labels],
                          dtype='S20')
        hf.create_dataset(name='gain_db',
                          data=gain_db,
                          dtype=np.float32)

    hf.close()

    print('Write out hdf5 file to {}'.format(hdf5_path))
    print('Time spent: {} s'.format(time.time() - calculate_time))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--subdir', type=str, required=True)
    parser_logmel.add_argument('--data_type', type=str, required=True,
                               choices=['development', 'leaderboard', 'evaluation'])
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--mini_data', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'logmel':

        calculate_multi_features(args)

    else:
        raise Exception('Incorrect arguments!')
