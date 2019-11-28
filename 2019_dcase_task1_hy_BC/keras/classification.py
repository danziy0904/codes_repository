import numpy as np
import h5py
from sklearn.svm import SVC
import os
import sys
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utilities import calculate_accuracy, calculate_confusion_matrix, plot_confusion_matrix2, print_accuracy
import config as cfg


def prepare_data(datatype):
    workspace = os.path.join(os.path.expanduser('~'), "Downloads/dcase2019_task1")
    truncation_dir = os.path.join(workspace, 'features', 'truncation',
                                  'holdout_fold={}'.format(1))
    if datatype == 'train':
        hf = h5py.File(os.path.join(truncation_dir, 'train_hpss_lrad_15500.h5'), 'r')
        features = hf['feature'][:]
        targets = hf['target'][:]
        return features, np.argmax(targets, axis=-1)
    elif datatype == 'validate':
        hf = h5py.File(os.path.join(truncation_dir, 'validate_hpss_lrad_15500.h5'), 'r')
        features = hf['feature'][:]
        targets = hf['target'][:]
        return features, np.argmax(targets, axis=-1)


def model_validate(classifier, class_wise_accuracy=False, plot_confusion_matrix=False):
    x_val, y_val = prepare_data(datatype='validate')
    if class_wise_accuracy:
        predict = classifier.predict_proba(x_val)
        print(predict.shape)
        # predict = np.argmax(predict,axis=-1)
        # if plot_confusion_matrix:
        #     cm = calculate_confusion_matrix(y_val, predict, 10)
        #     plot_confusion_matrix2(cm, "svm", cfg.labels)
        #
        # print(y_val.shape)
        hf = h5py.File('leader-hpss_lrad-6m-svm.h5', 'w')
        hf.create_dataset(name='outputs', data=predict)
        hf.create_dataset(name='targets', data=y_val)
        hf.close()
        class_wise_accuracy = calculate_accuracy(y_val, predict, 10)
        print_accuracy(class_wise_accuracy, cfg.labels)
        score = np.mean(class_wise_accuracy)
    else:

        score = classifier.score(x_val, y_val)
        print('The accuracy of validation: {:.4f}'.format(score))
    return score


def train_trainset_and_predict_testset():
    c1 = [x * 0.1 for x in range(1, 11, 1)]
    c2 = [x * 1e-4 for x in range(1, 10, 1)]
    max_acc = 0
    for c3 in c1:
        for c4 in c2:
            print(c3, c4)
            classifier = SVC(C=c3, gamma=c4, random_state=10)
            X_train, y_train = prepare_data(datatype='train')
            classifier.fit(X_train, y_train)
            score = model_validate(classifier)
            max_acc = max(max_acc, score)

    print('max score：{:.4f}'.format(max_acc))


def train_dev_and_predict_learderboard():
    classifier = SVC(C=0.2, gamma=2e-4, random_state=10, probability=True)
    workspace = os.path.join(os.path.expanduser('~'), "Downloads/dcase2019_task1")
    truncation_dir = os.path.join(workspace, 'features', 'truncation',
                                  'holdout_fold={}'.format(1))
    hf = h5py.File(os.path.join(truncation_dir, 'dev_full_hpss.h5'), 'r')
    features = hf['feature'][:]
    targets = hf['target'][:]
    classifier.fit(features, np.argmax(targets, axis=-1))

    hf = h5py.File(os.path.join(truncation_dir, 'leader_hpss-6m.h5'), 'r')

    predict = classifier.predict_log_proba(hf['feature'][:])

    hf = h5py.File('leader-hpss-6m-svm.h5', 'w')
    hf.create_dataset(name='outputs', data=predict)
    hf.create_dataset(name='targets', data=targets)
    hf.close()


def train_validation_random_forest():
    base_path = '/home/r506/Downloads/dcase2019_task1/features/truncation/holdout_fold=1'
    train_hf = h5py.File(os.path.join(base_path, 'train_hpss_lrad_15500.h5'), 'r')
    validate_hf = h5py.File(os.path.join(base_path, 'validate_hpss_lrad_15500.h5'))
    feature = train_hf['feature'][:]
    target = train_hf['target'][:]
    target = np.argmax(target, axis=-1)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=10)
    param_grid = {'max_depth': [x for x in range(2, 10)], }
    gv_search = GridSearchCV(estimator=classifier, param_grid=param_grid,
                             scoring=make_scorer(precision_score, average="macro"), cv=5, n_jobs=4,
                             iid=True)
    gv_search.fit(feature, target)
    print(gv_search.cv_results_)
    print(gv_search.best_params_)
    print(gv_search.best_score_)


def randorm_forest_predict():
    base_path = '/home/r506/Downloads/dcase2019_task1/features/truncation/holdout_fold=1'
    train_hf = h5py.File(os.path.join(base_path, 'train_hpss_lrad_15500.h5'), 'r')
    validate_hf = h5py.File(os.path.join(base_path, 'validate_hpss_lrad_15500.h5'))
    t_feature = train_hf['feature'][:]
    t_target = np.argmax(train_hf['target'][:], axis=1)
    feature = validate_hf['feature'][:]
    target = validate_hf['target'][:]
    target = np.argmax(target, axis=1)
    classifier = RandomForestClassifier(n_estimators=2000, max_depth=3, random_state=10, n_jobs=4)
    # score = classifier.score(feature, np.argmax(target, axis=1))
    # print('The accuracy of validation: {:.4f}'.format(score))
    param_grid = {'max_features': [3], }
    gv_search = GridSearchCV(estimator=classifier, param_grid=param_grid,
                             scoring=make_scorer(precision_score, average="macro"), cv=5, n_jobs=4,
                             iid=True)
    gv_search.fit(t_feature, t_target)
    a = gv_search.score(feature, target)
    print(a)
    print(gv_search.cv_results_)
    print(gv_search.best_params_)
    print(gv_search.best_score_)


if __name__ == '__main__':
    # train_trainset_and_predict_testset()
    randorm_forest_predict()
