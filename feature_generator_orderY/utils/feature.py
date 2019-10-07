import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
import pandas as pd
import h5py
import os
import time
import argparse

from utilities import read_audio,creat_folder
import config


# DATASET_DIR="D:\Learning\science_deeplearning\learning_code\learning\Dcase_data"
# WORKSPACE="D:\Learning\science_deeplearning\learning_code\learning\workspace"
# DEV_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-development"


class LogMelExtractor():#梅尔声音特征提取
    def __init__(self,sample_rate,window_size,overlap,mel_bin):
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np. hamming(window_size)#这是个一维数组，含有window_size列


        self.melW = librosa.filters.mel(sr = sample_rate,
                                        #n_fft = 2048,
                                        n_fft=window_size,
                                        n_mels = mel_bin,
                                        fmin = 0,
                                        #fmax = sample_rate//2
                                        fmax = 24000
                                        ).T
        """(window_size,mel_bin)"""
    def transform(self,audio):
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap

        [f,t,x] = signal.spectrogram(
            audio,
            window=ham_win,#分窗大小
            nperseg=window_size,#int型，窗口长度
            noverlap= overlap,#int型，窗口与窗口的重叠面积
            detrend= False,
            return_onesided=True,
            mode= 'magnitude'
        )

        #seq_len在这个输出shape可以得到，时间轴帧数
        #返回f为频域信息，t为时域信息，x为时频图,x的横轴代表帧数，每个点对应一个帧 纵轴代表每个帧经过傅立叶变换后得到的能量系数
        x = x.T
        print(x.shape,"3")
        x = np.dot(x,self.melW)
        print(x.shape,"4")
        x = np.log(x+1e-8)
        print(x.shape)
        x = x.astype(np.float32)
        print(x.shape)
        x = x.T
        return x;
    """(mel_bin,time_index)--->(200,499)"""
def calculate_logmel(audio_path,sample_rate,feature_extractor):
    (audio,fs)=read_audio(audio_path,target_fs=sample_rate)
    feature=feature_extractor .transform(audio)
    return feature

# feature_extractor = LogMelExtractor(sample_rate=44100,
#                                     window_size=2048,
#                                     overlap=672,
#                                     mel_bin=64)
# feature = calculate_logmel(audio_path="D:/Learning/science_deeplearning/learning_code/learning/Dcase_data/TUT-urban-acoustic-scenes-2018-development/airport-barcelona-0-0-a.wav",
#                            sample_rate=44100,
#                            feature_extractor=feature_extractor)

#print(feature)
# print(feature.shape)
# plt.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
# plt.show()
#特征写入h5文件
# seq_len = config.seq_len
# mel_bins = config.mel_bins
# def  write_h5py(h5py_path,feature):
#     hf = h5py.File(h5py_path,'w')
#     hf.create_dataset(
#         name = 'feature',
#         data = feature,
#         shape=( seq_len, mel_bins),
#         # shape=(0,seq_len,mel_bins),
#         # maxshape=(None,seq_len,mel_bins),
#         dtype=np.float32
#     )
#     # hf['feature'].resize((1,seq_len,mel_bins))
#     # hf['feature'][1]=feature
# f ="D:/Learning/science_deeplearning/learning_code/learning/Dcase_data/TUT-urban-acoustic-scenes-2018-development/11.hdf5"
# write_h5py(f,feature)
def read_development_meta(meta_csv):#读取音频文件名

    df=pd.read_csv(meta_csv,sep='\t')
    df=pd.DataFrame(df)

    audio_names = []
    scene_labels = []
    identifiers = []
    source_labels = []

    for row in df.iterrows():
        audio_name = row[1]['filename'].split('/')[1]
        scene_label = row[1]['scene_label']
        identifier = row[1]['identifier']
        source_label = row[1]['source_label']
        #曾今出错的地方，需要分别audio_name,scene_label,identifier,identifier,source_label叠成一个列表
        audio_names.append(audio_name)
        scene_labels.append(scene_label)
        identifiers.append(identifier)
        source_labels.append(source_label)
    return audio_names,scene_labels,identifiers,source_labels

def calculate_features(args):
    #批量读取数据提取特征？
    dataset_dir = args.dataset_dir
    subdir = args.subdir#我猜应该是TUT-urban-acoustic-scenes-2018-development？
    data_type  = args.data_type#数据集是测试集验证集
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins

    #音频所在的路径
    audio_dir = os.path.join(dataset_dir,subdir,'audio')
    #csv文件所在
    if data_type =='development':
        meta_csv = os.path.join(dataset_dir,subdir,'meta.csv')
    #未完待续，这里可能还有验证集

    #不是很懂mini_data的用处
    #反正这里是创建装特征的h5文件
    if mini_data:
        hdf5_path = os.path.join(workspace,'features','logmel',subdir,
                                 'mini_{}.h5'.format(data_type))
    else:
        hdf5_path = os.path.join(workspace,'features','logmel',subdir,
                                 '{}.h5'.format(data_type))
    creat_folder(os.path.dirname(hdf5_path))

    #特征提取
    feature_extractor = LogMelExtractor(sample_rate = sample_rate,
                                        window_size = window_size,
                                        overlap = overlap,
                                        mel_bin = mel_bins)
    #读取csv文件
    if data_type == 'development':
        [audio_names,scene_labels,identifiers,source_labels]= read_development_meta(meta_csv)

    #这特么还是呀mini_data
    if mini_data:
        audio_num = 300#打乱音频顺序抽取300个音频
        random_state = np.random.RandomState(0)
        audio_indexes = np.arange(len(audio_names))
        random_state.shuffle(audio_indexes)
        audio_indexes = audio_indexes[0:audio_num]

        if data_type =='development':
            scene_labels = [scene_labels[idx] for idx in audio_indexes]
            identifiers = [identifiers[idx] for idx in audio_indexes]
            source_labels = [source_labels[idx] for idx in audio_indexes]
    print('Number of audios:{}'.format(len(audio_names)))

    #开始创建hdf5文件来存储特征
    hf = h5py.File(hdf5_path,'w')

    hf.create_dataset(
        name = 'feature',
        shape = (0,1,mel_bins,499),
        maxshape = (None,1,mel_bins,499),
        dtype = np.float32)#三维的目的为了放索引？

    calculate_time = time.time()
    print("这个地方注意一下！1")

    for(n,audio_name) in enumerate(audio_names):#这里花了点时间调试，注意for循环的关系，只有音频循环提取特征
        print("有问题吗")
        print(n,audio_name)

        #批量处理特征
        audio_path = os.path.join(audio_dir,audio_name)

        feature = calculate_logmel(audio_path = audio_path,
                                   sample_rate = sample_rate,
                                   feature_extractor = feature_extractor
                                   )
        print(feature.shape)
        print("这个点注意一下！2")

        hf['feature'].resize((n + 1,1, mel_bins, 499))
        hf['feature'][n] = feature

        if False:
            plt.matshow(feature.T,origin = 'lower',aspect = 'auto',cmap ='jet')
            plt.show()

        #将文件名标签之类的信息写到h5文件中
    hf.create_dataset(
            name = 'filename',
            data = [s.encode() for s in audio_name],
            dtype = 'S50'
        )
    if data_type == 'development':
         hf.create_dataset(
                name = 'scene_label',
                data = [s.encode() for s in scene_labels ],
                dtype = 'S20'
            )
         hf.create_dataset(
                name = 'identifier',
                data = [s.encode() for s in identifiers],
                dtype = 'S20'
            )
         hf.create_dataset(name = 'source_label',
                             data = [s.encode() for s in source_labels],
                             dtype = 'S20'
                             )

    hf.close()
    print('Write out hdf5 file to {}'.format(hdf5_path))
    print ('Time spent:{} s'.format(time.time()-calculate_time))

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

        calculate_features(args)

    else:
        raise Exception('Incorrect arguments!')



























