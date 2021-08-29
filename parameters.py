
# parameters.py

import os

# class Dataset:#
#     name = 'Fer2013'
#     train_folder = 'fer2013_features/Training'
#     validation_folder = 'fer2013_features/PublicTest'
#     test_folder = 'fer2013_features/PrivateTest'
#     trunc_trainset_to = -1
#     trunc_validationset_to = -1
#     trunc_testset_to = -1

class Dataset:#
    name = 'jaffe'
    train_folder = 'jaffe_features/Training'
    validation_folder = 'jaffe_features/PublicTest'
    test_folder = 'jaffe_features/PrivateTest'
    trunc_trainset_to = -1
    trunc_validationset_to = -1
    trunc_testset_to = -1


class Hyperparams:
    random_state = 0
    epochs = 10000
    kernel = 'rbf'  # 'rbf', 'linear', 'poly' or 'sigmoid'
    decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'
    features = "landmarks_and_lbp" 

 
class Training:
    save_model = True
    save_model_path = "saved_model.bin"

DATASET = Dataset()
TRAINING = Training()
HYPERPARAMS = Hyperparams()