import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_generator import DataGenerator
from utilities import (creat_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy,
                       plot_confusion_matrix, print_accuracy,
                       )
from Baseline import move_data_to_gpu, BaselineCnn
import config

Model = BaselineCnn
batch_size = 64
# workspace = '/home/r506/hhy/workspace'
subdir = 'TUT-urban-acoustic-scenes-2018-development'
# logging.getLogger().setLevel(logging.INFO)
#logging.basicConfig(level=logging.INFO,filename='train.log')
workspace = '/home/r506/hhy/workspace'
dataset_dir  = '/home/r506/DCase/data'

def evaluate(model, generator, data_type, devices, max_iteration, cuda):
    """Evaluate

    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
      max_iteration: int, maximum iteration for validation
      cuda: bool.

    Returns:
      accuracy: float
    """

    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                devices=devices,
                                                shuffle=True,
                                                max_iteration=max_iteration)

    # Forward
    dict = forward(model=model,
                   generate_func=generate_func,
                   cuda=cuda,
                   return_target=True)

    outputs = dict['output']  # (audios_num, classes_num)
    #print(outputs)
    targets = dict['target']  # (audios_num, classes_num)
    #print(targets)
    predictions = np.argmax(outputs, axis=-1)  # (audios_num,)返回最大值的行索引

    # Evaluate
    classes_num = outputs.shape[-1]  # 变成一维？
    #print(classes_num) --->输出 10

    loss = F.nll_loss(torch.Tensor(outputs), torch.LongTensor(targets)).numpy()
    loss = float(loss)

    confusion_matrix = calculate_confusion_matrix(
        targets, predictions, classes_num)

    accuracy = calculate_accuracy(targets, predictions, classes_num,
                                  average='macro')  # 计算准确率是计算整个模型的准确率

    return accuracy, loss  # 计算损失函数与准确率的模块


def forward(model, generate_func, cuda, return_target):
    """Forward data to a model.

    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool

    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """

    outputs = []
    audio_names = []

    if return_target:
        targets = []

    # Evaluate on mini-batch
    for data in generate_func:

        if return_target:
            (batch_x, batch_y, batch_audio_names) = data

            #print(batch_y)
            # print(batch_audio_names)

        else:
            (batch_x, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()
        batch_output = model(batch_x)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)

        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)  # axis=0按行拼接
    dict['output'] = outputs  # 训练得到的结果
    # print(dict['output'])

    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names
    #print(dict['audio_name'])

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets  # 标签结果
        #print(dict['target'])

    return dict  # 将数据传给模型


def train(args):
    #workspace = args.workspace
    #dataset_dir = args.dataset_dir#为了知道验证集和测试集的csv
    #subdir = args.subdir#中介的名字
    filename = args.filename
    validate = args.validate
    holdout_fold = args.holdout_fold
    #mini_data = args.mini_data
    # cuda = args.cuda#这个一般是返回ture或者false
    cuda=True



    labels = config.labels


    devices = ['a']

    classes_num = len(labels)
    #classes_num = 10


    #hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,'development.h5')


    hdf5_path = os.path.join(workspace,'features', 'logmel',subdir, 'development.h5')


    #"/home/r506/hhy/workspace"

    # print(hdf5_path_1)
    # print(hdf5_path_2)
    # print(hdf5_path_3)
    # Paths
    # if mini_data:
    #     hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
    #                              'mini_development.h5')
    # else:
    #     hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
    #                              'development.h5')

    if validate:#这里开始分成训练集与验证集

        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                     'fold{}_train.txt'.format(holdout_fold))

        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                        'fold{}_evaluate.txt'.format(holdout_fold))

        models_dir = os.path.join(workspace, 'models', subdir, filename,
                                  'holdout_fold={}'.format(holdout_fold))

    else:#否则在全部数据集上训练
        dev_train_csv = None
        dev_validate_csv = None

        models_dir = os.path.join(workspace, 'models', subdir,
                              'full_train')

    creat_folder(models_dir)

    # Model
    model = Model(classes_num)


    model.cuda()

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    train_bgn_time = time.time()

    # Train on mini batches
    for (iteration, (batch_x, batch_y)) in enumerate(generator.generate_train()):

        # Evaluate
        if iteration % 100 == 0:

            train_fin_time = time.time()

            (tr_acc, tr_loss) = evaluate(model=model,
                                         generator=generator,
                                         data_type='train',
                                         devices=devices,
                                         max_iteration=None,
                                         cuda=cuda)

            logging.info('tr_acc: {:.3f}, tr_loss: {:.3f}'.format(
                tr_acc, tr_loss))


            if validate:

                (va_acc, va_loss) = evaluate(model=model,
                                             generator=generator,
                                             data_type='validate',
                                             devices=devices,
                                             max_iteration=None,
                                             cuda=cuda)

                logging.info('va_acc: {:.3f}, va_loss: {:.3f}'.format(
                    va_acc, va_loss))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))

        # Reduce learning rate
        if iteration % 200 == 0 > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # Train
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        model.train()
        batch_output = model(batch_x)
        loss = F.nll_loss(batch_output, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 10000:
            break


def inference_validation_data(args):
    # Arugments & parameters
    # dataset_dir = args.dataset_dir
    # subdir = args.subdir
    # workspace = args.workspace
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    filename = args.filename
    cuda = True

    labels = config.labels


    devices = ['a']#只完成任务一的a

    validation = True
    classes_num = len(labels)

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                             'development.h5')

    dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                 'fold1_train.txt')

    dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold{}_evaluate.txt'.format(holdout_fold))

    model_path = os.path.join(workspace, 'models', subdir, filename,
                              'holdout_fold={}'.format(holdout_fold),
                              'md_{}_iters.tar'.format(iteration))

    # Load model
    model = Model(classes_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Predict & evaluate
    for device in devices:
        print('Device: {}'.format(device))

        # Data generator
        generator = DataGenerator(hdf5_path=hdf5_path,
                                  batch_size=batch_size,
                                  dev_train_csv=dev_train_csv,
                                  dev_validate_csv=dev_validate_csv)

        generate_func = generator.generate_validate(data_type='validate',
                                                    devices=device,
                                                    shuffle=False)

        # Inference
        dict = forward(model=model,
                       generate_func=generate_func,
                       cuda=cuda,
                       return_target=True)

        outputs = dict['output']  # (audios_num, classes_num)
        targets = dict['target']  # (audios_num, classes_num)

        predictions = np.argmax(outputs, axis=-1)

        classes_num = outputs.shape[-1]  # 获取classes_num的值

        # Evaluate
        confusion_matrix = calculate_confusion_matrix(
            targets, predictions, classes_num)

        class_wise_accuracy = calculate_accuracy(targets, predictions,
                                                 classes_num)

        # Print
        print_accuracy(class_wise_accuracy, labels)
        print('confusion_matrix: \n', confusion_matrix)

        # Plot confusion matrix
        plot_confusion_matrix(
            confusion_matrix,
            title='Device {}'.format(device.upper()),
            labels=labels,
            values=class_wise_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    #parser_train.add_argument('--dataset_dir', type=str, required=True)
    #parser_train.add_argument('--subdir', type=str, required=True)
    #parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--holdout_fold', type=int)
    # parser_train.add_argument('--cuda', action='store_true', default=False)
    #parser_train.add_argument('--mini_data', action='store_true', default=False)

    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    #parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    #parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    #parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--holdout_fold', type=int, required=True)
    parser_inference_validation_data.add_argument('--iteration', type=int, required=True)
    #parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)
    else:
        raise Exception('Error argument!')