import re
import matplotlib.pyplot as plt
import os
import numpy as np


def load_log_file(path):
    """
    正则匹配出所有相关数据并返回
    :param path:
    :return:
    """
    with open(path, mode='r') as file:
        train_loss, test_loss, train_acc, test_acc, epochs = [], [], [], [], []
        iter = []
        n = 0
        for line in file.readlines():
            line = line.strip()
            pattern1 = re.compile(r'tr_acc: ([\d.]+), tr_loss: ([\d.]+)')
            for item in re.findall(pattern1, line):
                if len(item):
                    train_acc.append(float(item[0]))
                    train_loss.append(float(item[1]))
                    if n % 100 == 0:
                        iter.append(n)
                    n += 100

            pattern2 = re.compile(r'va_acc: ([\d.]+), va_loss: ([\d.]+)')
            for item in re.findall(pattern2, line):
                test_acc.append(float(item[0]))
                test_loss.append(float(item[1]))

            pattern3 = re.compile(
                r'epoch: ([\d.]+)/([\d.]+) iteration: ([\d.]+), train time: ([\d.]+) s, validate time: ([\d.]+) s')
            for item in re.findall(pattern3, line):
                epochs.append(float(item[0]))
    return train_loss, train_acc, test_loss, test_acc, iter, epochs


def plot_experiment_with_iteration(log_path):
    train_loss, train_acc, test_loss, test_acc, iter, epochs = load_log_file(log_path)

    max_acc = np.max(test_acc)
    max_acc_iter = iter[np.argmax(test_acc)]
    print(max_acc)
    print(max_acc_iter)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplot(121)
    ## 标注最大值的点
    plt.scatter([max_acc_iter, ], [max_acc, ], s=50, color='b', zorder=1)
    ##注释
    plt.annotate('({},{})'.format(max_acc_iter, max_acc), xy=(max_acc_iter, max_acc), xycoords='data',
                 xytext=(-60, -60),
                 textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    # 垂线
    plt.plot([max_acc_iter, max_acc_iter, ], [0, max_acc, ], 'k--', linewidth=2.5)
    p1 = plt.plot(iter, train_acc, '.--', color='#6495ED', zorder=0)
    p2 = plt.plot(iter, test_acc, '.--', color='#FF6347', zorder=0)
    plt.legend([p1[0], p2[0]], ['train_acc', 'test_acc'])
    plt.title('train/test acc')
    plt.xlabel('iteration')
    plt.ylabel('acc')
    plt.ylim((0, 1))
    plt.xscale('log')

    plt.subplot(122)
    p1 = plt.plot(iter, train_loss, '.--', color='#6495ED')
    p2 = plt.plot(iter, test_loss, '.--', color='#FF6347')
    plt.legend([p1[0], p2[0]], ['train_loss', 'test_loss'])
    plt.title('train/test loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.xscale('log')

    fig.tight_layout()  ## 这个很好用 可以防止规范子图的显示范围
    plt.show()


def plot_experiment_with_epoch(log_path):
    train_loss, train_acc, test_loss, test_acc, iter, epochs = load_log_file(log_path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplot(121)
    print(epochs)
    p1 = plt.plot(epochs, train_acc, '-', color='#6495ED', zorder=0)
    p2 = plt.plot(epochs, test_acc, '-', color='#FF6347', zorder=0)
    plt.legend([p1[0], p2[0]], ['train_acc', 'test_acc'])
    plt.title('train/test acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.ylim((0, 1))

    plt.subplot(122)
    p1 = plt.plot(epochs, train_loss, '-', color='#6495ED')
    p2 = plt.plot(epochs, test_loss, '-', color='#FF6347')
    plt.legend([p1[0], p2[0]], ['train_loss', 'test_loss'])
    plt.title('train/test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    fig.tight_layout()  ## 这个很好用 可以防止规范子图的显示范围
    plt.show()


def show_log_curve(index):
    workspace = "/home/r506/Downloads/dcase2019_task1"
    log_path = os.path.join(workspace, 'logs', 'main_keras', '{}.log'.format(index))
    plot_experiment_with_iteration(log_path)
    # plot_lr_range_test(log_path)


def plot_lr_range_test(path):
    lrs, acc = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            pattern = re.compile(r'epoch: ([\d.]+) lr: ([\d.]+) lr_tr_acc: ([\d.]+)')
            for item in re.findall(pattern, line):
                lrs.append(float(item[1]))
                acc.append(float(item[2]))

    lrs.pop(0)
    acc.pop(0)
    print(lrs)
    print(acc)
    # plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, 0.004, 8), acc, )
    # plt.yticks(np.linspace(0, 1, 9))
    # plt.ylim((0., 1.))
    # plt.xlim((-1, 2))
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    # 0022 74.7
    show_log_curve('0258')
