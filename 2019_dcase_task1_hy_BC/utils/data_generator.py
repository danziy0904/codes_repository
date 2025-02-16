import numpy as np
import h5py
import csv
import time
import logging

from utilities import calculate_scalar, scale
import config as config
import keras
import sys
import os
from data_augmentation import get_random_eraser

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))


class DataGenerator(object):

    def __init__(self, hdf5_path, batch_size, dev_train_csv=None,
                 dev_validate_csv=None, seed=1234):
        """
        Inputs:
          hdf5_path: str
          batch_size: int
          dev_train_csv: str | None, if None then use all data for training
          dev_validate_csv: str | None, if None then use all data for training
          seed: int, random seed
        """

        self.batch_size = batch_size

        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        lb_to_ix = config.lb_to_ix

        # Load data
        load_time = time.time()
        logging.info(hdf5_path)
        hf = h5py.File(hdf5_path, 'r')

        self.audio_names = np.array([s.decode() for s in hf['filename'][:]])
        self.x = hf['feature'][:]
        self.scene_labels = [s.decode() for s in hf['scene_label'][:]]
        self.identifiers = [s.decode() for s in hf['identifier'][:]]
        self.source_labels = [s.decode() for s in hf['source_label']]
        # self.y = np.array([lb_to_ix[lb] for lb in self.scene_labels])
        self.gain_db = hf['gain_db'][:]
        self.y = np.array(
            [keras.utils.to_categorical(lb_to_ix[lb], len(lb_to_ix)) for lb in
             self.scene_labels])  # one-hot encode

        hf.close()
        logging.info('Loading data time: {:.3f} s'.format(
            time.time() - load_time))

        # Use all data for training
        if dev_train_csv is None and dev_validate_csv is None:

            self.train_audio_indexes = np.arange(len(self.audio_names))
            logging.info('Use all development data for training. ')

        # Split data to training and validation
        else:

            self.train_audio_indexes = self.get_audio_indexes_from_csv(
                dev_train_csv)

            self.validate_audio_indexes = self.get_audio_indexes_from_csv(
                dev_validate_csv)

            logging.info('Split development data to {} training and {} '
                         'validation data. '.format(len(self.train_audio_indexes),
                                                    len(self.validate_audio_indexes)))

        # Calculate scalar
        (self.mean, self.std) = calculate_scalar(
            self.x[self.train_audio_indexes])

    def get_audio_indexes_from_csv(self, csv_file):
        """Calculate indexes from a csv file. 
        
        Args:
          csv_file: str, path of csv file
        """

        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            lis = list(reader)

        audio_indexes = []
        lis.pop(0)
        for li in lis:
            audio_name = li[0].split('/')[1]

            if audio_name in self.audio_names:
                audio_index = np.where(self.audio_names == audio_name)[0][0]
                audio_indexes.append(audio_index)

        return audio_indexes

    def generate_train(self, alpha=0.):
        """Generate mini-batch data for training.
        alpha: mixup alpha,0 means not use mixup

        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
        """

        batch_size = self.batch_size
        audio_indexes = np.array(self.train_audio_indexes)
        audios_num = len(audio_indexes)

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        # def func(x):
        #     N, C, H, W = x.shape
        #     xx = np.zeros((N, C, H + 1, W))
        #     xx[:, :, 0:431, :] = x
        #     xx[:, :, -1, :] = x[:, :, 0, :]
        #     return xx

        while True:

            # Reset pointer
            if pointer + batch_size > audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1
            batch_x = self.x[batch_audio_indexes]
            batch_x_gain_db = self.gain_db[batch_audio_indexes]  # 全部x的gain_db值
            batch_y = self.y[batch_audio_indexes]

            if alpha == 0.:
                yield self.transform(batch_x), batch_y
            else:
                # mixup implementation
                weight = np.random.beta(alpha, alpha, batch_size)
                x_weight = weight.reshape(batch_size, 1, 1, 1)
                y_weight = weight.reshape(batch_size, 1)
                index = np.random.permutation(batch_size)

                x1, x2 = batch_x, batch_x[index]
                #x = x1 * x_weight + x2 * (1 - x_weight)
                gain1, gain2 = batch_x_gain_db, batch_x_gain_db[index]
                x_weight = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - x_weight) / x_weight)
                x = ((x1 * x_weight + x2 * (1 - x_weight)) / np.sqrt(x_weight ** 2 + (1 - x_weight) ** 2))
                y1, y2 = batch_y, batch_y[index]
                y = y1 * y_weight + y2 * (1 - y_weight)
                x = self.transform(x)
                ### match with random erase with probabilities
                # x = [get_random_eraser(p=0.4, v_l=np.min(x1), v_h=np.max(x1))(x1) for x1 in x]

                # yield np.stack(x), y
                # x = [func(xx) for xx in x]
                yield x, y

    def generate_validate(self, data_type, devices, shuffle,
                          max_iteration=None):
        """Generate mini-batch data for evaluation.

        Args:
          data_type: 'train' | 'validate'
          devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
          max_iteration: int, maximum iteration for validation
          shuffle: bool

        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
          batch_audio_names: (batch_size,)
        """

        batch_size = self.batch_size

        if data_type == 'train':
            audio_indexes = np.array(self.train_audio_indexes)

        elif data_type == 'validate':
            audio_indexes = np.array(self.validate_audio_indexes)

        else:
            raise Exception('Invalid data_type!')

        if shuffle:
            self.validate_random_state.shuffle(audio_indexes)

        # Get indexes of specific devices
        devices_specific_indexes = []

        for n in range(len(audio_indexes)):
            if self.source_labels[audio_indexes[n]] in devices:
                devices_specific_indexes.append(audio_indexes[n])

        logging.info('Number of {} audios in specific devices {}: {}'.format(
            data_type, devices, len(devices_specific_indexes)))

        audios_num = len(devices_specific_indexes)

        iteration = 0
        pointer = 0

        while True:

            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = devices_specific_indexes[
                                  pointer: pointer + batch_size]

            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]
            batch_audio_names = self.audio_names[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_audio_names

    def transform(self, x):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, self.mean, self.std)


class TestDataGenerator(DataGenerator):

    def __init__(self, dev_hdf5_path, test_hdf5_path, batch_size):
        """Data generator for test data. 
        
        Inputs:
          dev_hdf5_path: str
          test_hdf5_path: str
          batch_size: int
        """

        super(TestDataGenerator, self).__init__(
            hdf5_path=dev_hdf5_path,
            batch_size=batch_size,
            dev_train_csv=None,
            dev_validate_csv=None)

        # Load test data
        load_time = time.time()
        hf = h5py.File(test_hdf5_path, 'r')

        self.test_audio_names = np.array(
            [s.decode() for s in hf['filename'][:]])

        self.test_x = hf['feature'][:]

        hf.close()

        logging.info('Loading data time: {:.3f} s'.format(
            time.time() - load_time))

    def generate_test(self):

        audios_num = len(self.test_x)
        audio_indexes = np.arange(audios_num)
        batch_size = self.batch_size

        pointer = 0

        while True:

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]

            pointer += batch_size

            batch_x = self.test_x[batch_audio_indexes]
            batch_audio_names = self.test_audio_names[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_audio_names
