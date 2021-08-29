# data_loader.py

from parameters import DATASET, HYPERPARAMS
import numpy as np
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize


def load_data(validation=False, test=False):
    
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == "Fer2013":
        # load train set
        if HYPERPARAMS.features == "landmarks_and_hog":
            data_dict['X'] = np.load(DATASET.train_folder + '/landmarks.npy')
            print(data_dict['X'].shape)
            data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
            print(data_dict['X'].shape)
            data_dict['X'] = np.concatenate((data_dict['X'], np.load(DATASET.train_folder + '/hog_features.npy')), axis=1)
            print(len(data_dict['X']))
            print(data_dict['X'].shape)

        elif HYPERPARAMS.features == "landmarks_and_lbp":
            data_dict['X'] = np.load(DATASET.train_folder + '/landmarks.npy')
            print('in data_loader \n', data_dict['X'].shape)
            data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
            print(data_dict['X'].shape)
            # data_dict['X'] = np.concatenate((data_dict['X'], np.load(DATASET.train_folder + '/lbp_features.npy')), axis=1)
            data_dict['X'] = np.concatenate((np.load(DATASET.train_folder + '/lbp_features.npy'), data_dict['X']), axis=1)

            print(data_dict['X'].shape)

            # data_dict['X'] = np.load(DATASET.train_folder + '/lbp_features.npy')

        elif HYPERPARAMS.features == "landmarks":
            data_dict['X'] = np.load(DATASET.train_folder + '/landmarks.npy')
            data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
        elif HYPERPARAMS.features == "hog":
            data_dict['X'] = np.load(DATASET.train_folder + '/hog_features.npy')
        else:
            print("Error '{}' features not recognized".format(HYPERPARAMS.features))
        data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')
        if DATASET.trunc_trainset_to > 0:
            data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :]
            data_dict['Y'] = data_dict['Y'][0:DATASET.trunc_trainset_to]
        if validation:
            # load validation set 
            if HYPERPARAMS.features == "landmarks_and_hog":
                validation_dict['X'] = np.load(DATASET.validation_folder + '/landmarks.npy')
                validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
                validation_dict['X'] = np.concatenate((validation_dict['X'], np.load(DATASET.validation_folder + '/hog_features.npy')), axis=1)

            elif HYPERPARAMS.features == "landmarks_and_lbp":
                validation_dict['X'] = np.load(DATASET.validation_folder + '/landmarks.npy')
                validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
                # validation_dict['X'] = np.concatenate((validation_dict['X'], np.load(DATASET.validation_folder + '/lbp_features.npy')), axis=1)
                validation_dict['X'] = np.concatenate((np.load(DATASET.validation_folder + '/lbp_features.npy'), validation_dict['X']), axis=1)
                # validation_dict['X'] = np.load(DATASET.validation_folder + '/lbp_features.npy')

            elif HYPERPARAMS.features == "landmarks":
                validation_dict['X'] = np.load(DATASET.validation_folder + '/landmarks.npy')
                validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
            elif HYPERPARAMS.features == "hog":
                validation_dict['X'] = np.load(DATASET.validation_folder + '/hog_features.npy')
            else:
                print("Error '{}' features not recognized".format(HYPERPARAMS.features))
            validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')
            if DATASET.trunc_validationset_to > 0:
                validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :]
                validation_dict['Y'] = validation_dict['Y'][0:DATASET.trunc_validationset_to]
        if test:
            # load train set
            if HYPERPARAMS.features == "landmarks_and_hog":
                test_dict['X'] = np.load(DATASET.test_folder + '/landmarks.npy')
                test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
                test_dict['X'] = np.concatenate((test_dict['X'], np.load(DATASET.test_folder + '/hog_features.npy')), axis=1)

            elif HYPERPARAMS.features == "landmarks_and_lbp":
                test_dict['X'] = np.load(DATASET.test_folder + '/landmarks.npy')
                print('in data_loader->test', test_dict['X'].shape)
                test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
                print(test_dict['X'].shape)
                # test_dict['X'] = np.concatenate((test_dict['X'], np.load(DATASET.test_folder + '/lbp_features.npy')), axis=1)
                test_dict['X'] = np.concatenate((np.load(DATASET.test_folder + '/lbp_features.npy'), test_dict['X']),axis=1)
                print(test_dict['X'].shape)

                # test_dict['X'] = np.load(DATASET.test_folder + '/lbp_features.npy')

            elif HYPERPARAMS.features == "landmarks":
                test_dict['X'] = np.load(DATASET.test_folder + '/landmarks.npy')
                test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
            elif HYPERPARAMS.features == "hog":
                test_dict['X'] = np.load(DATASET.test_folder + '/hog_features.npy')
            else:
                print("Error '{}' features not recognized".format(HYPERPARAMS.features))
            test_dict['Y'] = np.load(DATASET.test_folder + '/labels.npy')
            np.save(DATASET.test_folder + "/lab.npy", test_dict['Y'])
            if DATASET.trunc_testset_to > 0:
                test_dict['X'] = test_dict['X'][0:DATASET.trunc_testset_to, :]
                test_dict['Y'] = test_dict['Y'][0:DATASET.trunc_testset_to]

        if not validation and not test:
            return data_dict
        elif not test:
            return data_dict, validation_dict
        else: 
            return data_dict, validation_dict, test_dict
    else:
        print("Unknown dataset")
        exit()
