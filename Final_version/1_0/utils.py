import joblib as jb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from sklearn.metrics import  f1_score, roc_auc_score, precision_recall_curve, confusion_matrix


def create_path(PATH, list_approachs):
    preds_folder = ["preds_train1", "preds_val1", "preds_test"]
    for pred in preds_folder:
        if pred not in os.listdir(PATH):
            os.mkdir(os.path.join(PATH, pred))
        
        path_pred = os.path.join(PATH, pred)
    
        for approach in list_approachs:
            apc_path = f"approach{approach}"
            if apc_path not in os.listdir(path_pred):
                os.mkdir(os.path.join(path_pred, apc_path))

def save_y_id(PATH, approach, y_train1, y_valid, test_id):
    name_ytrain1 = PATH + "/preds_train1/approach{}/y.pkl.z".format(approach) 
    jb.dump(y_train1, name_ytrain1)
    name_yvalid = PATH + "/preds_val1/approach{}/y.pkl.z".format(approach) 
    jb.dump(y_valid, name_yvalid)
    name_id = PATH + "/preds_test/approach{}/id.pkl.z".format(approach) 
    jb.dump(test_id, name_id)

def evaluate(y_pred, y_true, plot_matrix=False, roc=False):
    if roc:
        roc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if plot_matrix:
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.matshow(conf_matrix, cmap=plt.cm.gray)
        plt.show()
    if roc:
        return roc, f1
    
    return 0, f1


def better_threshold(y_true, y_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_best = 0
    threshold = 0
    for i in range(len(precisions)):
        if precisions[i] != 0 and recalls[i] != 0:
            f1 = 2*(precisions[i]*recalls[i])/(precisions[i] + recalls[i])
        else:
            f1 = 0
        if f1 > f1_best:
            f1_best = f1
            threshold = thresholds[i]
            
    
    return threshold, f1_best

# Validations
def gen_strat_folds(df, tgt_name=None, n_splits=5, shuffle=True, random_state=42, isSparse=False):
    
    if isSparse == False:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (_, valid_index) in enumerate(skf.split(df.drop(columns=tgt_name), df[tgt_name])):
            df.loc[df[df.index.isin(valid_index)].index, 'fold'] = fold

    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        folds = np.zeros(df.shape[0])
        for fold, (_, valid_index) in enumerate(skf.split(df.tocsr()[:, :-1], df.tocsr()[:, -1].A)):
            folds[valid_index] = fold
        
        df = sparse.hstack((df, folds[:,None]))
    
    return df
# Functions to Neural Nets

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    loss = f1_loss(y_true, y_pred)
    return 1 - loss

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, tf.float32), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), tf.float32), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, tf.float32), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), tf.float32), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.grid(True)
    plt.show()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_common_columns(paths, train_col, valid_col, test_col):
    train_path, valid_path, test_path = paths
    common = []
    for col in train_col:
        file_name = col[len(train_path):]
        if valid_path + file_name in valid_col and test_path + file_name in test_col:
            common.append(file_name)

    return common

def get_data(path):
  predictions = []
  names = []
  for file_name in os.listdir(path):
    if file_name.endswith(".z"):
      names.append("{}/{}".format(path, file_name))
      predictions.append(jb.load(names[-1]))
 
  return names, predictions

def transform_1d_array(data):
  array = []
  for idx in range(data.shape[0]):
    array.append(data[idx][0])
 
  return np.array(array)

def make_dataframe(names, data):
  array = np.zeros((data[0].shape[0], len(data)))
  for idx, pred in enumerate(data):
    if len(pred.shape) == 2:
      pred = transform_1d_array(pred)
    if len(pred) == data[0].shape[0]:
      array[:, idx] = pred
 
  return pd.DataFrame(data=array, columns=names)

def return_dataframe(paths, approach):
    df = []
    for path in paths:
        path_ = path + str(approach)
        names, preds = get_data(path_)
        df.append(make_dataframe(names, preds))

    return df

def return_params_level_2(path):
    train_path = "{}preds_train1/approach".format(path)
    valid_path = "{}preds_val1/approach".format(path)
    test_path = "{}preds_test/approach".format(path)
    approachs = list(range(1, 8))

    return train_path, valid_path, test_path, approachs

def add_y_and_id(paths, dfs, approach):
    train_path, valid_path, test_path = paths
    X_train, X_valid, X_test = dfs
    name_ytrain1 = "{}{}/y.pkl.z".format(train_path, approach)
    if name_ytrain1 in X_train.columns:
        X_train.drop(name_ytrain1, axis=1, inplace=True)
    X_train['y'] = jb.load(name_ytrain1)
    name_yvalid = "{}{}/y.pkl.z".format(valid_path, approach)
    if name_yvalid in X_valid.columns:
        X_valid.drop(name_yvalid, axis=1, inplace=True)
    X_valid['y'] = jb.load(name_yvalid)
    name_id = "{}{}/id.pkl.z".format(test_path, approach)
    if name_id in X_test.columns:
        X_test.drop(name_id, axis=1, inplace=True)
    X_test['id'] = jb.load(name_id).astype(np.int)

    return X_train, X_valid, X_test

def read_dataframes(paths, approach):
    train_path, valid_path, test_path = paths

    train = pd.read_csv(f"{train_path}{approach}/train.csv")
    valid = pd.read_csv(f"{valid_path}{approach}/valid.csv")
    test = pd.read_csv(f"{test_path}{approach}/test.csv")

    mask = get_common_columns(paths, train.columns, valid.columns, test.columns)
    X_train = train[[train_path + col for col in mask]]
    X_valid = valid[[valid_path + col for col in mask]]
    X_test = test[[test_path + col for col in mask]]
    X_train.columns = mask
    X_valid.columns = mask
    X_test.columns = mask
    name_ytrain1 = f"{approach}/y.pkl.z" 
    if name_ytrain1 in X_train.columns:
      X_train.drop(name_ytrain1, axis=1, inplace=True)
    name_yvalid = f"{approach}/y.pkl.z"
    if name_yvalid in X_valid.columns:
      X_valid.drop(name_yvalid, axis=1, inplace=True)
    name_id = f"{approach}/id.pkl.z"
    if name_id in X_test.columns:
      X_test.drop(name_id, axis=1, inplace=True)

    return X_train, X_valid, X_test

    