import os
import sys
import numpy as np
import argparse
import h5py
import time
import logging

import keras
import keras.backend as K
from datetime import datetime

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, plot_confusion_matrix2,
                       plot_confusion_matrix, print_accuracy, calculate_stats,
                       write_leaderboard_submission, write_evaluation_submission, compute_time_consumed)
from models_keras import resnet_50
import tensorflow as tf
import config

batch_size = 32
cf = tf.ConfigProto()
cf.gpu_options.allow_growth = True
sess = tf.Session(config=cf)
K.set_session(sess)

train_file = 'fold1_train.csv'
evaluate_file = 'fold1_evaluate.csv'


def evaluate(model, generator, data_type, devices, max_iteration, ):
    """Evaluate

    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
      max_iteration: int, maximum iteration for validation

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
                   return_target=True)

    outputs = dict['output']  # (audios_num, classes_num)
    print(outputs,"main_keras_59")
    targets = dict['target']  # (audios_num, classes_num)
    print(targets,"main_keras_61")
    predictions = np.argmax(outputs, axis=-1)  # (audios_num,)

    # Evaluate
    classes_num = outputs.shape[-1]

    # categorical_crossentropy 必须配合softmax使用 binary_crossentropy 配合sigmoid 使用
    loss = K.mean(keras.metrics.categorical_crossentropy(K.constant(targets), K.constant(outputs)))
    loss = K.eval(loss)

    # confusion_matrix = calculate_confusion_matrix(
    #     targets, predictions, classes_num)

    targets = np.argmax(targets, axis=-1)

    accuracy = calculate_accuracy(targets, predictions, classes_num,
                                  average='macro')
    return accuracy, loss


def forward(model, generate_func, return_target):
    """Forward data to a model.

    Args:
      generate_func: generate function
      return_target: bool

    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """

    outputs = []
    audio_names = []

    if return_target:
        targets = []

    # Evaluate on mini-batch
    # n = 0
    for data in generate_func:

        if return_target:
            (batch_x, batch_y, batch_audio_names) = data

        else:
            (batch_x, batch_audio_names) = data

        # Predict
        batch_output = model.predict(batch_x)

        # Append data
        outputs.append(batch_output)
        audio_names.append(batch_audio_names)

        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets

    return dict


def train(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    filename = args.filename
    validate = args.validate
    holdout_fold = args.holdout_fold
    mini_data = args.mini_data
    alpha = args.alpha

    model_arg = 'attention2'

    labels = config.labels
    seq_len = config.seq_len
    mel_bins = config.mel_bins

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'mini_development_2019.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'development_hpss_lrad_2019.h5')

    if validate:

        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                     train_file)

        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                        evaluate_file)

        models_dir = os.path.join(workspace, 'models', subdir, filename,
                                  'holdout_fold={}'.format(holdout_fold))

    else:
        dev_train_csv = None
        dev_validate_csv = None

        models_dir = os.path.join(workspace, 'models', subdir, filename,
                                  'full_train')

    create_folder(models_dir)

    model = resnet_50(4,seq_len, mel_bins, 10)

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=1e-3))

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv,
                              )

    train_bgn_time = time.time()

    # Train on mini batches
    start_time = datetime.now()
    max_acc = 0

    iters = 9185 // batch_size
    epochs = 60
    max_iteration = epochs * iters
    epoch = 0
    for (iteration, (batch_x, batch_y)) in enumerate(generator.generate_train(alpha=alpha)):

        # Evaluate
        if iteration % 100 == 0:

            train_fin_time = time.time()

            (tr_acc, tr_loss) = evaluate(model=model,
                                         generator=generator,
                                         data_type='train',
                                         devices=devices,
                                         max_iteration=None, )

            logging.info('tr_acc: {:.3f}, tr_loss: {:.3f}'.format(
                tr_acc, tr_loss))

            if validate:
                (va_acc, va_loss) = evaluate(model=model,
                                             generator=generator,
                                             data_type='validate',
                                             devices=devices,
                                             max_iteration=None, )

                if va_acc > max_acc:
                    max_acc = va_acc
                    if iteration >= 10000:
                        save_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        save_out_path = os.path.join(
                            models_dir, 'md_{}_iters_max_{}_{}.h5'.format(iteration, model_arg, save_time))

                        model.save(save_out_path)
                        logging.info('Model saved to {}'.format(save_out_path))
                logging.info('va_acc: {:.3f}, va_loss: {:.3f}, max_va_acc: {:.3f}'.format(
                    va_acc, va_loss, max_acc))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info('epoch: {}/{} '
                         'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                         ''.format(epoch, epochs, iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 8000:
            save_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters_max_{}_{}.h5'.format(iteration, model_arg, save_time))

            model.save(save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))

        #  Reduce learning rate
        if iteration % 300 == 0 and iteration > 0:
            old_lr = float(K.get_value(model.optimizer.lr))
            K.set_value(model.optimizer.lr, old_lr * 0.9)

        model.train_on_batch(batch_x, batch_y)
        if iteration == iters * epoch:
            epoch += 1

        # Stop learning
        if iteration == max_iteration + 1:
            compute_time_consumed(start_time)
            break


def create_feature_in_h5py(generator, layer_output, hf, data_type):
    train_generate_func = generator.generate_validate(data_type=data_type,
                                                      devices='a',
                                                      shuffle=False)

    # Inference
    outputs = []
    targets = []
    for data in train_generate_func:
        (batch_x, batch_y, batch_audio_names) = data

        # Predict
        if data_type == 'train':
            batch_output = layer_output([batch_x, 0])[0]
        else:
            batch_output = layer_output([batch_x, 0])[0]
        # Append data
        outputs.append(batch_output)
        targets.append(batch_y)

    outputs = np.concatenate(outputs, axis=0)
    print(outputs.shape)
    targets = np.concatenate(targets, axis=0)
    print(targets.shape)

    hf.create_dataset(name='target',
                      data=targets)
    hf.create_dataset(name='feature',
                      data=outputs)

    hf.close()


def inference_data_to_truncation(args):
    """
    截取训练集和验证集的倒2层模型输出, 用于其他分类器训练和分类使用
    :param args:
    :return:
    """
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    filename = args.filename
    # data_type = args.data_type?

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                             'development_hpss_lrad_2019.h5')

    dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                 train_file)

    dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    evaluate_file)

    # 保存截断特征
    truncation_dir = os.path.join(workspace, 'features', 'truncation',
                                  'holdout_fold={}'.format(holdout_fold))
    create_folder(truncation_dir)

    # model_path = os.path.join(workspace, 'models', subdir, filename,
    #                           'holdout_fold={}'.format(holdout_fold),
    #                           'md_{}_iters_max_attention2.h5'.format(iteration))

    base = '/home/r506/Downloads/dcase2019_task1/models/' \
           'TAU-urban-acoustic-scenes-2019-development/main_keras/model-6M'
    model_path = os.path.join(base, 'md_15500_iters_max_attention2_hpss_lrad_6M_184log.h5')

    hdf5_train_path = os.path.join(truncation_dir,
                                   'train_hpss_lrad_15500.h5')
    hdf5_validate_path = os.path.join(truncation_dir,
                                      'validate_hpss_lrad_15500.h5')
    train_hf = h5py.File(hdf5_train_path, 'w')
    validate_hf = h5py.File(hdf5_validate_path, 'w')

    # load model
    model = keras.models.load_model(model_path)

    layer_output = K.function([model.layers[0].input, K.learning_phase()],
                              [model.layers[-2].output])

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)

    create_feature_in_h5py(generator, layer_output, train_hf, data_type='train')
    create_feature_in_h5py(generator, layer_output, validate_hf, data_type='validate')


def inference_validation_data(args):
    """
    使用验证集或训练集推断模型
    :param args:
    :return:
    """
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    filename = args.filename

    # data_type = args.data_type

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                             'development_hpss_lrad_2019.h5')

    dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                 'fold{}_train.csv'.format(holdout_fold))

    dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold{}_evaluate.csv'.format(holdout_fold))

    model_path = os.path.join(workspace, 'models', subdir, filename,
                               'holdout_fold={}'.format(holdout_fold),
                              #'model-6M',
                              #'md_15500_iters_max_attention2_hpss_lrad_6M_184log.h5'
                              'md_17000_iters_max_attention2_2019-08-23 11:08:47.h5')

    model = keras.models.load_model(model_path)

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
                       return_target=True)

        outputs = dict['output']  # (audios_num, classes_num)
        targets = dict['target']  # (audios_num, classes_num)

        hf = h5py.File('keras/development_hpss_lrad_2019.h5')
        hf.create_dataset(name='outputs', data=outputs)
        hf.create_dataset(name='targets',
                          data=targets)

        hf.close()
        # 多分类交叉熵
        predictions = np.argmax(outputs, axis=-1)

        classes_num = outputs.shape[-1]

        # Evaluate
        targets = np.argmax(targets, axis=-1)
        confusion_matrix = calculate_confusion_matrix(
            targets, predictions, classes_num)

        class_wise_accuracy = calculate_accuracy(targets, predictions,
                                                 classes_num)

        # Print
        print_accuracy(class_wise_accuracy, labels)
        print('confusion_matrix: \n', confusion_matrix)

        # Plot confusion matrix
        np.save('data5', confusion_matrix)
        plot_confusion_matrix2(
            confusion_matrix,
            title='Device {}'.format(device.upper()),
            labels=labels, )


def inference_leaderboard_data(args):
    """
    使用排行榜数据集, 推断模型结果
    :param args:
    :return:
    """
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    dev_subdir = args.dev_subdir
    leaderboard_subdir = args.leaderboard_subdir
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename

    labels = config.labels
    ix_to_lb = config.ix_to_lb
    subdir = args.subdir

    classes_num = len(labels)

    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', dev_subdir,
                                 'development_hpss_lrad_2019.h5')

    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', leaderboard_subdir,
                                  'development_hpss_lrad_2019.h5')

    model_path = os.path.join(workspace, 'models', dev_subdir, filename,
                              'full_train',
                              'md_17000_iters_hpss.h5')
    print(model_path)

    submission_path = os.path.join(workspace, 'submissions', leaderboard_subdir,
                                   filename, 'iteration={}'.format(iteration),
                                   'submission1.csv')

    create_folder(os.path.dirname(submission_path))

    # Load model
    model = keras.models.load_model(model_path)

    # Data generator
    generator = TestDataGenerator(dev_hdf5_path=dev_hdf5_path,
                                  test_hdf5_path=test_hdf5_path,
                                  batch_size=batch_size)

    generate_func = generator.generate_test()

    # Predict
    dict = forward(model=model,
                   generate_func=generate_func,
                   return_target=True)

    audio_names = dict['audio_name']  # (audios_num,)
    outputs = dict['output']
    # (audios_num, classes_num)

    predictions = np.argmax(outputs, axis=-1)  # (audios_num,)
    # Write result to submission csv
    write_leaderboard_submission(submission_path, audio_names, predictions)


def inference_full_train_truncation_data(args):
    """
    截取整个开发集送入模型的倒二层输出
    :param args:
    :return:
    """
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    dev_subdir = args.dev_subdir
    # leaderboard_subdir = args.leaderboard_subdir
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename

    labels = config.labels
    ix_to_lb = config.ix_to_lb
    # subdir = args.subdir

    classes_num = len(labels)

    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', dev_subdir,
                                 'development_lr_2019.h5')

    model_path = os.path.join(workspace, 'models', dev_subdir, filename,
                              'leader-model-6m',
                              'md_16000_iters_max_attention2_lr_203log.h5')
    print(model_path)

    # Load model
    model = keras.models.load_model(model_path)

    #
    generator = DataGenerator(hdf5_path=dev_hdf5_path,
                              batch_size=batch_size,
                              dev_train_csv=None,
                              dev_validate_csv=None)

    generate_func = generator.generate_validate(data_type='train',
                                                devices='a',
                                                shuffle=False)
    layer_output = K.function([model.layers[0].input, K.learning_phase()],
                              [model.layers[-2].output])
    # Predict
    outputs = []
    targets = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_y, _) = data

        # Predict
        batch_output = layer_output([batch_x, 0])[0]

        # Append data
        outputs.append(batch_output)
        targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    batch_y = np.concatenate(targets, axis=0)
    dict['targets'] = batch_y

    # (audios_num, classes_num)
    base = '/home/r506/Downloads/dcase2019_task1/features/truncation/holdout_fold=1'
    hf = h5py.File(os.path.join(base, 'leader_full_lr.h5'), 'w')
    hf.create_dataset(name='feature',
                      data=dict['output'])
    hf.create_dataset(name='target',
                      data=dict['targets'])

    hf.close()


def create_leaderboard_feature_in_h5py(generator, layer_output, hf):
    train_generate_func = generator.generate_test()

    # Inference
    outputs = []
    targets = []
    for data in train_generate_func:
        (batch_x, batch_audio_name) = data
        # Pedict
        batch_output = layer_output([batch_x, 0])[0]
        # Append data
        outputs.append(batch_output)
        targets.append(batch_audio_name)

    outputs = np.concatenate(outputs, axis=0)
    print(outputs.shape)
    targets = np.concatenate(targets, axis=0)
    print(targets.shape)

    hf.create_dataset(name='feature',
                      data=outputs)
    hf.create_dataset(name='target',
                      data=[s.encode() for s in targets], dtype='S10')

    hf.close()


def inference_leaderboard_truncation_data(args):
    """
    截取排行榜数据集送入模型的倒二层输出
    :param args:
    :return:
    """
    # Arugments & parameters
    dev_subdir = args.dev_subdir
    leaderboard_subdir = args.leaderboard_subdir
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    holdout_fold = args.holdout_fold

    labels = config.labels

    truncation_dir = os.path.join(workspace, 'features', 'truncation',
                                  'holdout_fold={}'.format(holdout_fold))
    create_folder(truncation_dir)

    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', dev_subdir,
                                 'development_lr_2019.h5')

    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', leaderboard_subdir,
                                  'leaderboard_lr_2019.h5')

    # model_path = os.path.join(workspace, 'models', dev_subdir, filename,
    #                           'full_train',
    base = '/home/r506/Downloads/dcase2019_task1/models/' \
           'TAU-urban-acoustic-scenes-2019-development/main_keras/leader-model-6m'
    model_path = os.path.join(base, 'md_16000_iters_max_attention2_lr_203log.h5')
    # print(model_path)

    # Load model
    model = keras.models.load_model(model_path)
    # print(model.input_shape)

    # Data generator
    generator = TestDataGenerator(dev_hdf5_path=dev_hdf5_path,
                                  test_hdf5_path=test_hdf5_path,
                                  batch_size=batch_size)

    layer_output = K.function([model.layers[0].input, K.learning_phase()],
                              [model.layers[-2].output])

    hdf5_leaderboard_path = os.path.join(truncation_dir,
                                         'leader_lr-6m.h5')
    leaderboard_hf = h5py.File(hdf5_leaderboard_path, 'w')

    create_leaderboard_feature_in_h5py(generator, layer_output, leaderboard_hf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--subdir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--holdout_fold', type=int)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--alpha', type=float, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    parser_inference_validation_data = subparsers.add_parser('inference_data_to_truncation')
    parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--holdout_fold', type=int, required=True)
    parser_inference_validation_data.add_argument('--iteration', type=int, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)
    # parser_inference_validation_data.add_argument('--data_type', type=str, required=True)

    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--holdout_fold', type=int, required=True)
    parser_inference_validation_data.add_argument('--iteration', type=int, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)

    parser_inference_leaderboard_data = subparsers.add_parser('inference_leaderboard_data')
    parser_inference_leaderboard_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--dev_subdir', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--leaderboard_subdir', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--workspace', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--iteration', type=int, required=True)
    parser_inference_leaderboard_data.add_argument('--cuda', action='store_true', default=False)

    parser_inference_leaderboard_data = subparsers.add_parser('inference_leaderboard_truncation_data')
    parser_inference_leaderboard_data.add_argument('--holdout_fold', type=int, required=True)
    parser_inference_leaderboard_data.add_argument('--dev_subdir', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--leaderboard_subdir', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--workspace', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--iteration', type=int, required=True)
    parser_inference_leaderboard_data.add_argument('--cuda', action='store_true', default=False)

    parser_inference_evaluation_data = subparsers.add_parser('inference_development_truncation_data')
    parser_inference_evaluation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--dev_subdir', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--iteration', type=int, required=True)
    parser_inference_evaluation_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    elif args.mode == 'inference_leaderboard_data':
        inference_leaderboard_data(args)

    elif args.mode == 'inference_data_to_truncation':
        inference_data_to_truncation(args)
    elif args.mode == 'inference_leaderboard_truncation_data':
        inference_leaderboard_truncation_data(args)
    else:
        raise Exception('Error argument!')
