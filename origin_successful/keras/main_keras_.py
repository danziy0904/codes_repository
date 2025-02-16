import os
import sys
import numpy as np
import argparse
import h5py
import time
import logging

import keras
import keras.backend.tensorflow_backend as K
from datetime import datetime

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, plot_confusion_matrix2,
                       plot_confusion_matrix, print_accuracy, calculate_stats,
                       write_leaderboard_submission, write_evaluation_submission, compute_time_consumed)
# from models_keras import BaselineCnn, Vggish, Vggish_single_attention, Vggish_two_attentionFPN, Vggish_two_attention, \
#     Vggish_single_attention_MF, Vggish_attention_no_fcn, Vggish_3attentionn
# from CLR import CyclicLR
import tensorflow as tf
import config
from datetime import datetime
from earlystop import EarlyStopping
from sub_net import subsp_net
from ord_net import ord_net
from keras.models import model_from_json

# model = model.build(input_shape=(3, 320, 64))  # 72.0 0130.log
# model = Vggish(431, 84, 10)  # 71.3 0032.log
# model = Vggish_two_attention2(320, 64, 10)  # 71.3 0032.log
batch_size = 32
cf = tf.compat.v1.ConfigProto()  # 记录设备指派情况
cf.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cf)  # 动态申请显存
K.set_session(sess)
# train_file = 'fold1_train_new.txt'
# evaluate_file = 'fold1_validate.txt'

train_file = 'fold1_train.txt'
evaluate_file = 'fold1_evaluate.txt'


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
    # if data_type == 'train':
    generate_func = generator.generate_validate(data_type=data_type,
                                                devices=devices,
                                                shuffle=True,
                                                max_iteration=max_iteration)

    # Forward
    dict = forward(model=model,
                   generate_func=generate_func,
                   return_target=True)

    outputs = dict['output']  # (audios_num, classes_num)
    # print(outputs)
    # print(outputs.shape,"main_keras_71")
    targets = dict['target']  # (audios_num, classes_num)
    # print(targets)
    # print(targets.shape,"main_keras_74")
    predictions = np.argmax(outputs, axis=-1)  # (audios_num,)

    # Evaluate
    classes_num = outputs.shape[-1]
    # print(classes_num,"main_keras_76")

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
    a = 0
    for data in generate_func:

        if return_target:
            (batch_x, batch_y, batch_audio_names) = data

        else:
            (batch_x, batch_audio_names) = data

        # print("Pros starts here!")
        # print(batch_x)
        # print(len(batch_x))#32
        # print(batch_x.shape)#(32,2,64,320)

        # Predict
        #model.summary()
        batch_output = model.predict(batch_x)
        # a = a+1
        # print(a)a为完成一个Epoch需要的batch——size数
        # print(batch_output,"main_keras_122")
        # a  = a + len(batch_output)
        # print(a,"main_keras_124")
        # print("Pros ends?")
        # Append data
        outputs.append(batch_output)
        audio_names.append(batch_audio_names)

        if return_target:
            targets.append(batch_y)

    dict = {}
    # print("快出现问题了！")
    # 问题所在
    # batch_x = np.concatenate(batch_x, axis=0)
    # print(batch_x)说明输入特征尺度是统一的
    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    # 下面的代码可以运行
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

    model_arg = args.model

    labels = config.labels
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    classes_num = len(labels)

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'mini_development.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'development_fold1_train.h5')

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

    # Model
    # if model_arg == 'attention1':
    #     model = Vggish_single_attention(320, 64, 10)
    #     logging.info("loading Vggish_single_attention")
    # elif model_arg == 'attention2':
    #     model = Vggish_two_attention(320, 64, 10)
    #     logging.info("loading Vggish_two_attention")
    # elif model_arg == 'attention3':
    #     model = Vggish_3attentionn(320, 64, 10)
    #     logging.info("loading Vggish_3attention")
    # elif model_arg == 'attention5':
    #     model = Vggish_single_attention_MF(320, 64, 10)
    #     logging.info("loading Vggish_single_attention_MF")

    model = ord_net(2, 64, 320,"channels_first")
    logging.info("loading ord_net")
    # else:
    #     model = Vggish_attention_no_fcn(320, 64, 10)
    #     logging.info("loading Vggish_2attention_no_fcn")

    #model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(1e-3))

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv,
                              )

    train_bgn_time = time.time()

    # Train on mini batches
    start_time = datetime.now()
    # clr = CyclicLR(model=model, base_lr=0.0001, max_lr=0.0005,
    #                step_size=5000., mode='triangular')
    # clr.on_train_begin()
    max_iteration = 1000
    max_acc = 0

    # earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, )
    # earlyStop.on_train_begin()

    iters = 6122 // batch_size  # 完成全部数据（即一个epoch）batch-size数
    epochs = max_iteration // iters
    epochs += 1
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
            # evaluate只是前向传播 然后打印出当前的准确度和误差

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
                    # if iteration >= 5000:
                    save_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # save_out_path_model = os.path.join(
                    #     models_dir, 'md_{}_iters_max_{}_{}_model.json'.format(iteration, model_arg, save_time))
                    # save_out_path_weights = os.path.join(
                    #     models_dir, 'md_{}_iters_max_{}_{}_weights.json'.format(iteration, model_arg, save_time))
                    #
                    save_out_path_model = os.path.join(models_dir, 'subnet_model.json')
                    save_out_path_weights = os.path.join(models_dir, 'subnet_weights.h5')
                    # model.save(save_out_path)#保存了结构和参数
                    # 保存模型和参数
                    model_json = model.to_json()
                    with open(save_out_path_model, "w") as json_file:
                        json_file.write(model_json)
                    model.save_weights(save_out_path_weights)

                    logging.info('Model saved to {}'.format(save_out_path_model))
                    logging.info('Model saved to {}'.format(save_out_path_weights))

            logging.info('va_acc: {:.3f}, va_loss: {:.3f}, max_va_acc: {:.3f}'.format(
                va_acc, va_loss, max_acc))

        train_time = train_fin_time - train_bgn_time
        validate_time = time.time() - train_fin_time

        logging.info('epoch: {}/{} '
                     'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                     ''.format(epoch, epochs, iteration, train_time, validate_time))

        logging.info('------------------------------------')

        train_bgn_time = time.time()

        # # Save model
        # if iteration % 1000 == 0 and iteration > 0:
        #     save_out_path = os.path.join(
        #         models_dir, 'md_{}_iters.h5'.format(iteration))
        #
        #     model.save(save_out_path)
        #     logging.info('Model saved to {}'.format(save_out_path))

        # # # Reduce learning rate
        if iteration % 200 == 0 and iteration > 0:
            old_lr = float(K.get_value(model.optimizer.lr))
            K.set_value(model.optimizer.lr, old_lr * 0.9)
        #
        model.train_on_batch(batch_x, batch_y)
        # train_on_batch 是进行一次前向传播和反向传播然后参数变化 那才是真正的意义上的训练
        # clr.on_batch_end()
        if iteration == iters * epoch:
            epoch += 1
        #     earlyStop.on_epoch_end(epoch, logs={'val_loss': va_loss})

        # Stop learning
        if iteration == max_iteration + 1:
            compute_time_consumed(start_time)
            # earlyStop.on_train_end()
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
                             'development_hpss_lrad.h5')

    dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                 train_file)

    dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    evaluate_file)

    # 保存截断特征
    truncation_dir = os.path.join(workspace, 'features', 'truncation',
                                  'holdout_fold={}'.format(holdout_fold))
    create_folder(truncation_dir)

    model_path = os.path.join(workspace, 'models', subdir, filename,
                              'holdout_fold={}'.format(holdout_fold),
                              'md_{}_iters_max_attention2_2019-05-31 00:48:09.h5'.format(iteration))

    # model_path = os.path.join(workspace, 'appendixes',
    #                           'md_{}_iters_max_76.2_Vggish_two_attention.h5'.format(iteration))

    hdf5_train_path = os.path.join(truncation_dir,
                                   'train_hpss_l+r_6900.h5')
    hdf5_validate_path = os.path.join(truncation_dir,
                                      'validate_hpss_l+r_6900.h5')
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
                             'development_fold1_train.h5')

    # hdf5_path = os.path.join(
    # '/home/r506/Downloads/dcase2018_task1-master/features/logmel/TUT-urban-acoustic-scenes-2018-development/development_hpss_lrad.h5')
    dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                 'fold{}_train.txt'.format(holdout_fold))

    dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold{}_evaluate.txt'.format(holdout_fold))

    # model_path = os.path.join(workspace, 'models', subdir, filename,
    #                           'holdout_fold={}'.format(holdout_fold),
    #                           'md_6600_iters_max_subsp_net_2019-10-21 10:51:47.h5')
    model_path = os.path.join(
        '/home/r506/hhy/workspace/models/TUT-urban-acoustic-scenes-2018-development/main_keras/holdout_fold=1/md_400_iters_max_subsp_net_2019-10-21 21:05:11.h5')
    #
    # model_path = '/home/r506/Downloads/dcase2018_task1-master/models/' \
    #              'TUT-urban-acoustic-scenes-2018-development/main_keras/' \
    #              'new/DAN-DFF/md_9700_iters_max_attention2_78.1.h5'
    # model_path = "/home/r506/hhy/workspace/models/TUT-urban-acoustic-scenes-2018-development/main_keras/holdout_fold=1/md_6600_iters_max_subsp_net_2019-10-21 10:51:47.h5"
    # model = keras.models.load_model(model_path)
    ##载入模型和参数
    json_file = open(
        '/home/r506/hhy/workspace/models/TUT-urban-acoustic-scenes-2018-development/main_keras/holdout_fold=1/subnet_model.json')
    model_file = json_file.read()  # 读模型
    json_file.close()
    model = model_from_json(model_file)
    model.load_weights(
        '/home/r506/hhy/workspace/models/TUT-urban-acoustic-scenes-2018-development/main_keras/holdout_fold=1/subnet_weights.h5')
    print("load the model from disk!")

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
        # plot_confusion_matrix(
        #     confusion_matrix,
        #     title='Device {}'.format(device.upper()),
        #     labels=labels,
        #     values=class_wise_accuracy)
        plot_confusion_matrix2(
            confusion_matrix,
            title='The best performance of the proposed DAN-DFF method',
            labels=labels, )


def inference_leaderboard_data(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    dev_subdir = args.dev_subdir
    leaderboard_subdir = args.leaderboard_subdir
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename

    labels = config.labels
    ix_to_lb = config.ix_to_lb
    # subdir = args.subdir

    classes_num = len(labels)

    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', dev_subdir,
                                 'development.h5')

    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', leaderboard_subdir,
                                  'leaderboard_hpss.h5')

    model_path = os.path.join(workspace, 'models', dev_subdir, filename,
                              'full_train',
                              'md_{}_iters.h5'.format(iteration))
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
                   return_target=False)

    audio_names = dict['audio_name']  # (audios_num,)
    outputs = dict['output']  # (audios_num, classes_num)

    predictions = np.argmax(outputs, axis=-1)  # (audios_num,)

    # Write result to submission csv
    write_leaderboard_submission(submission_path, audio_names, predictions)


def inference_evaluation_data(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    dev_subdir = args.dev_subdir
    eval_subdir = args.eval_subdir
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename

    labels = config.labels
    ix_to_lb = config.ix_to_lb

    classes_num = len(labels)

    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', dev_subdir,
                                 'development.h5')

    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', eval_subdir,
                                  'evaluation.h5')

    model_path = os.path.join(workspace, 'models', dev_subdir, filename,
                              'full_train',
                              'md_{}_iters.h5'.format(iteration))

    submission_path = os.path.join(workspace, 'submissions', eval_subdir,
                                   filename, 'iteration={}'.format(iteration),
                                   'submission.csv')

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
                   return_target=False)

    audio_names = dict['audio_name']  # (audios_num,)
    outputs = dict['output']  # (audios_num, classes_num)

    predictions = np.argmax(outputs, axis=-1)  # (audios_num,)

    # Write result to submission csv
    f = open(submission_path, 'w')

    write_evaluation_submission(submission_path, audio_names, predictions)


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
    parser_train.add_argument('--model', type=str)

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

    parser_inference_evaluation_data = subparsers.add_parser('inference_evaluation_data')
    parser_inference_evaluation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--dev_subdir', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--eval_subdir', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--iteration', type=int, required=True)
    parser_inference_evaluation_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    # logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    logs_dir = os.path.join("/home/r506/hhy/workspace/logs", args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    elif args.mode == 'inference_leaderboard_data':
        inference_leaderboard_data(args)

    elif args.mode == 'inference_evaluation_data':
        inference_evaluation_data(args)
    elif args.mode == 'inference_data_to_truncation':
        inference_data_to_truncation(args)

    else:
        raise Exception('Error argument!')
