import h5py
import numpy as np
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utilities import write_leaderboard_submission, calculate_accuracy


def validate_model_fusion():
    ###
    base = '/home/r506/2019_task1/'
    model1_path = os.path.join(base, 'keras/lr-6m-svm.h5')
    model2_path = os.path.join(base, 'keras/hpss-6m-svm.h5')
    model3_path = os.path.join(base, 'keras/hpss_lrad-6m-svm.h5')
    hf1 = h5py.File(model1_path, 'r')
    hf2 = h5py.File(model2_path, 'r')
    hf3 = h5py.File(model3_path, 'r')

    prediction1 = hf1['outputs'][:]
    prediction2 = hf2['outputs'][:]
    targets = hf3['targets'][:]
    prediction3 = hf3['outputs'][:]
    predictions = prediction2 + prediction3 + prediction1
    predictions = predictions / 3
    predictions = np.argmax(np.array(predictions), axis=1)
    a = calculate_accuracy(targets, predictions, 10)
    print(np.mean(a))


def leaderboard_model_fusion():
    base_path = '/home/r506/2019_task1/keras'
    model1_path = os.path.join(base_path,
                               'leader-hpss_lrad-6m-svm.h5')
    model2_path = os.path.join(base_path,
                               'leader-hpss-6m-svm.h5')
    model3_path = os.path.join(base_path,
                               'leader-lr-6m-svm.h5')
    hf1 = h5py.File(model1_path, 'r')
    hf2 = h5py.File(model2_path, 'r')
    hf3 = h5py.File(model3_path, 'r')

    prediction1 = hf1['outputs'][:]
    prediction2 = hf2['outputs'][:]
    prediction3 = hf3['outputs'][:]
    predictions = prediction2 + prediction3 + prediction1
    predictions = predictions / 3
    predictions = np.argmax(predictions, axis=1)
    targets = [str(i) + '.wav' for i in range(1200)]
    write_leaderboard_submission('/home/r506/submission1.csv', targets, predictions)


if __name__ == '__main__':
    leaderboard_model_fusion()
