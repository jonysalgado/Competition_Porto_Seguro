import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, precision_recall_curve, confusion_matrix
from tensorflow.keras import backend as K
from tensorflow import keras
import joblib as jb
from sklearn.linear_model import LogisticRegression
from tensorflow.python.keras.utils import metrics_utils
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from itertools import combinations
from sklearn.model_selection import train_test_split

PATH = "/content/drive/MyDrive/Colab Notebooks/Porto_Seguro_competition/"

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def remove_feature(list_remove, columns):
    return [x for x in columns if x not in list_remove]

def evaluate(y_pred, y_true, plot_matrix=True):
    score = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if plot_matrix:
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.matshow(conf_matrix, cmap=plt.cm.gray)
        plt.show()
    return score, f1

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, thrh):
    plt.figure(figsize=(8, 4))
    plt.axis([0, 1.1, 0, 1])
    precision_by_thrs = precisions[np.argmax(thresholds == thrh)]
    recall_by_thrs = recalls[np.argmax(thresholds == thrh)]
    
    plt.plot([thrh, thrh], [0., precision_by_thrs], "r:")
    plt.plot([thrh, thrh], [0., recall_by_thrs], "r:")
    plt.plot([0, thrh], [precision_by_thrs, precision_by_thrs], "r:")
    plt.plot([0, thrh], [recall_by_thrs, recall_by_thrs], "r:")
    plt.plot([thrh], [precision_by_thrs], "ro")        
    plt.plot([thrh], [recall_by_thrs], "ro")   
    
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")    
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(loc="center right", fontsize=14)
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.grid(True)
    
def better_threshold(precisions, recalls, thresholds):
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

def save_y_and_id(approach, y_train1, y_valid, test_id):
    name_ytrain1 = PATH + "./preds_train1/approach{}/y.pkl.z".format(approach) 
    jb.dump(y_train1, name_ytrain1)
    name_yvalid = PATH + "./preds_val1/approach{}/y.pkl.z".format(approach) 
    jb.dump(y_valid, name_yvalid)
    name_id = PATH + "./preds_test/approach{}/id.pkl.z".format(approach) 
    jb.dump(test_id, name_id)


# Tune Models
# ===================================================================

class TuneModels:
    def __init__(self, approach, train0, train1, valid, test):
        self.approach = approach
        self.X_train0, self.y_train0 = train0
        self.X_train1, self.y_train1 = train1
        self.valid, self.y_valid = valid
        self.test = test

    def tune_nn(self, params):

        hidden1, hidden2, epoch, learning_rate = params
        model = keras.models.Sequential([
            keras.Input(shape=self.X_train0.shape[1], sparse=True),
            keras.layers.Dense(hidden1, activation="relu"),
            keras.layers.Dense(hidden2, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(loss=f1_loss,
                    optimizer=keras.optimizers.Adam(lr=learning_rate),
                    metrics=[f1_m,precision_m, recall_m])
            
        with tf.device(get_available_gpus()[0]):
            model.fit(self.X_train0, 
                        self.y_train0, 
                        epochs=epoch, 
                        validation_data=(self.valid, self.y_valid),
                        verbose=1)
        p = model.predict(self.X_train1)
        y_pred = (p >= 0.5).astype(int)

        model_name_train1 = PATH + "./preds_train1/approach{}/nn_{}_{}_{}_{}.pkl.z".format(self.approach,hidden1, hidden2, epoch, learning_rate) 
        jb.dump(y_pred, model_name_train1)

        p = model.predict(self.valid)
        y_pred = (p >= 0.5).astype(int)
        _, metric = evaluate(y_pred, self.y_valid, plot_matrix=False)
        model_name_val1 = PATH + "/preds_val1/approach{}/nn_{}_{}_{}_{}.pkl.z".format(self.approach,hidden1, hidden2, epoch, learning_rate) 
        jb.dump(p, model_name_val1)
            
        p = model.predict(self.test)
        y_pred = (p >= 0.5).astype(int)
        model_name_test = PATH + "./preds_test/approach{}/nn_{}_{}_{}_{}.pkl.z".format(self.approach,hidden1, hidden2, epoch, learning_rate) 
        jb.dump(p, model_name_test)
        print(params, metric)
        print()
        return -metric

    def tune_logistic(self, params):
        tol, max_iter, C = params
        clf = LogisticRegression(random_state=42, 
                            solver='liblinear', 
                            max_iter=max_iter, 
                            tol=tol,
                            C=C,
                            penalty='l1')

        clf.fit(self.X_train0, self.y_train0)
        
        y_pred = clf.predict_proba(self.X_train1)[:, 1]
        model_name_train1 = PATH + "./preds_train1/approach{}/lr_{}_{}_{}.pkl.z".format(self.approach, tol, max_iter, C ) 
        jb.dump(y_pred, model_name_train1)
        
        y_pred = clf.predict_proba(self.valid)[:, 1]
        model_name_val1 = PATH + "/preds_val1/approach{}/lr_{}_{}_{}.pkl.z".format(self.approach, tol, max_iter, C ) 
        jb.dump(y_pred, model_name_val1)

        precisions, recalls, thresholds = precision_recall_curve(self.y_valid, y_pred)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (y_pred >= thrs).astype(int)
        _, metric = evaluate(y_pred, self.y_valid, plot_matrix=False)
        
        y_pred = clf.predict_proba(self.test)[:, 1]
        model_name_test = PATH + "./preds_test/approach{}/lr_{}_{}_{}.pkl.z".format(self.approach, tol, max_iter, C ) 
        jb.dump(y_pred, model_name_test)
        
        print(params, metric)
        print()
        
        return -metric

    def tune_lgbm(self, params):
        num_leaves, min_data_in_leaf, n_estimators, learning_rate = params
        mdl = LGBMClassifier(num_leaves=num_leaves,min_child_samples=min_data_in_leaf, learning_rate=learning_rate, 
                            n_estimators=n_estimators, random_state=42)
        mdl.fit(self.X_train0, self.y_train0)
        
        y_pred = mdl.predict_proba(self.X_train1)[:, 1]
        model_name_train1 = PATH + "/preds_train1/approach{}/lgbm_{}_{}_{}_{}.pkl.z".format(self.approach, num_leaves, min_data_in_leaf, n_estimators, learning_rate) 
        jb.dump(y_pred, model_name_train1)
        
        p = mdl.predict_proba(self.valid)[:,1]
        model_name_val1 = PATH + "/preds_val1/approach{}/lgbm_{}_{}_{}_{}.pkl.z".format(self.approach, num_leaves, min_data_in_leaf, n_estimators, learning_rate) 
        jb.dump(p, model_name_val1)

        precisions, recalls, thresholds = precision_recall_curve(self.y_valid, p)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (p >= thrs).astype(int)
        _, metric = evaluate(y_pred, self.y_valid, plot_matrix=False)
        
        p = mdl.predict_proba(self.test)[:,1]
        model_name_test = PATH + "/preds_test/approach{}/lgbm_{}_{}_{}_{}.pkl.z".format(self.approach, num_leaves, min_data_in_leaf, n_estimators, learning_rate) 
        jb.dump(p, model_name_test)
        
        print(params, metric)
        print()
        
        return -metric

    def tune_xgboost(self, params):
        learning_rate = params[0]
        max_depth = params[1]
        min_child_weight=params[2]
        subsample = params[3]
        colsample_bynode = params[4]
        num_parallel_tree = params[5]
        n_estimators = params[6]
        xgb = XGBClassifier(
                n_jobs=-1,
                eval_metric='auc',
                random_state=42,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bynode=colsample_bynode,
                num_parallel_tree=num_parallel_tree
        )
        fit_params = {
                'early_stopping_rounds': 100,
                'eval_metric' : 'auc',
                'eval_set': [(self.valid, self.y_valid)],
                'verbose': False,
            }
        xgb.fit(self.X_train0, self.y_train0, **fit_params)
        
        y_pred = xgb.predict_proba(self.X_train1)[:,1]
        model_name_train1 = PATH + "./preds_train1/approach{}/xgb_{}_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, learning_rate, max_depth, min_child_weight, subsample, colsample_bynode, num_parallel_tree, n_estimators) 
        jb.dump(y_pred, model_name_train1)
        
        p = xgb.predict_proba(self.valid)[:,1]
        model_name_val1 = PATH + "/preds_val1/approach{}/xgb_{}_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, learning_rate, max_depth, min_child_weight, subsample, colsample_bynode, num_parallel_tree, n_estimators)
        jb.dump(p, model_name_val1)
        
        precisions, recalls, thresholds = precision_recall_curve(self.y_valid, p)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (p >= thrs).astype(int)
        _, metric = evaluate(y_pred, self.y_valid, plot_matrix=False)
        p = xgb.predict_proba(self.test)[:,1]
        model_name_test = PATH + "./preds_test/approach{}/xgb_{}_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, learning_rate, max_depth, min_child_weight, subsample, colsample_bynode, num_parallel_tree, n_estimators)
        jb.dump(p, model_name_test)
        
        print(params, metric)
        print()
        
        return -metric

    def tune_trees(self, params):
        min_samples_leaf, weight, max_depth, n_estimators = params
        rf = RandomForestClassifier(n_estimators=n_estimators, 
                                    min_samples_leaf=min_samples_leaf, 
                                    max_depth = max_depth,
                                    class_weight = {0:weight, 1: 1},
                                    random_state=42)

        rf.fit(self.X_train0, self.y_train0)
        
        y_pred = rf.predict_proba(self.X_train1)[:,1]
        model_name_train1 = PATH + "./preds_train1/approach{}/rf_{}_{}_{}.pkl.z".format(self.approach, min_samples_leaf, weight, n_estimators) 
        jb.dump(y_pred, model_name_train1)
        
        precisions, recalls, thresholds = precision_recall_curve(self.y_train1, y_pred)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (y_pred >= thrs).astype(int)
        _, metric = evaluate(y_pred, self.y_train1, plot_matrix=False)
        
        p = rf.predict_proba(self.valid)[:, 1]
        model_name_val1 = PATH + "/preds_val1/approach{}/rf_{}_{}_{}.pkl.z".format(self.approach, min_samples_leaf, weight, n_estimators) 
        jb.dump(p, model_name_val1)
        
        p = rf.predict_proba(self.test)[:,1]
        model_name_test = PATH + "./preds_test/approach{}/rf_{}_{}_{}.pkl.z".format(self.approach, min_samples_leaf, weight, n_estimators) 
        jb.dump(p, model_name_test)
        
        print(params, metric)
        print()
        
        return -metric

    def tune_knn(self, params):
        n_neighbors = params[0]
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

        neigh.fit(self.X_train0, self.y_train0)
        
        y_pred = neigh.predict_proba(self.X_train1)[:,1]
        model_name_train1 = PATH + "./preds_train1/approach{}/knn_{}.pkl.z".format(self.approach, n_neighbors)
        jb.dump(y_pred, model_name_train1)
        
        precisions, recalls, thresholds = precision_recall_curve(self.y_train1, y_pred)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (y_pred >= thrs).astype(int)
        _, metric = evaluate(y_pred, self.y_train1, plot_matrix=False)
        
        p = neigh.predict_proba(self.valid)[:, 1]
        model_name_val1 = PATH + "/preds_val1/approach{}/knn_{}.pkl.z".format(self.approach, n_neighbors)
        jb.dump(p, model_name_val1)
        
        p = neigh.predict_proba(self.test)[:,1]
        model_name_test = PATH + "./preds_test/approach{}/knn_{}.pkl.z".format(self.approach, n_neighbors) 
        jb.dump(p, model_name_test)
        
        print(params, metric)
        print()
        
        return -metric

    def default_models(self):
        clf = LogisticRegression(random_state=42)

        clf.fit(self.X_train0, self.y_train0)
        
        y_pred = clf.predict_proba(self.X_train1)[:, 1]
        model_name_train1 = PATH + "./preds_train1/approach{}/lr_default.pkl.z".format(self.approach) 
        jb.dump(y_pred, model_name_train1)
        
        y_pred = clf.predict_proba(self.valid)[:, 1]
        model_name_val1 = PATH + "/preds_val1/approach{}/lr_default.pkl.z".format(self.approach) 
        jb.dump(y_pred, model_name_val1)

        precisions, recalls, thresholds = precision_recall_curve(self.y_valid, y_pred)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (y_pred >= thrs).astype(int)
        _, metric_clf = evaluate(y_pred, self.y_valid, plot_matrix=False)
        
        y_pred = clf.predict_proba(self.test)[:, 1]
        model_name_test = PATH + "./preds_test/approach{}/lr_default.pkl.z".format(self.approach) 
        jb.dump(y_pred, model_name_test)

        mdl = LGBMClassifier(random_state=42)
        mdl.fit(self.X_train0, self.y_train0)
            
        y_pred = mdl.predict_proba(self.X_train1)[:, 1]
        model_name_train1 = PATH + "/preds_train1/approach{}/lgbm_default.pkl.z".format(self.approach)
        jb.dump(y_pred, model_name_train1)
            
        p = mdl.predict_proba(self.valid)[:,1]
        model_name_val1 = PATH + "/preds_val1/approach{}/lgbm_default.pkl.z".format(self.approach)
        jb.dump(p, model_name_val1)

        precisions, recalls, thresholds = precision_recall_curve(self.y_valid, p)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (p >= thrs).astype(int)
        _, metric_lgbm = evaluate(y_pred, self.y_valid, plot_matrix=False)

        p = mdl.predict_proba(self.test)[:,1]
        model_name_test = PATH + "/preds_test/approach{}/lgbm_default.pkl.z".format(self.approach)
        jb.dump(p, model_name_test)


        mdl = XGBClassifier(n_jobs=-1, random_state=42)
        mdl.fit(self.X_train0, self.y_train0)
            
        y_pred = mdl.predict_proba(self.X_train1)[:, 1]
        model_name_train1 = PATH + "/preds_train1/approach{}/xgboost_default.pkl.z".format(self.approach)
        jb.dump(y_pred, model_name_train1)
            
        p = mdl.predict_proba(self.valid)[:,1]
        model_name_val1 = PATH + "/preds_val1/approach{}/xgboost_default.pkl.z".format(self.approach)
        jb.dump(p, model_name_val1)

        precisions, recalls, thresholds = precision_recall_curve(self.y_valid, p)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (p >= thrs).astype(int)
        _, metric_xgb = evaluate(y_pred, self.y_valid, plot_matrix=False)

        p = mdl.predict_proba(self.test)[:,1]
        model_name_test = PATH + "/preds_test/approach{}/xgboost_default.pkl.z".format(self.approach)
        jb.dump(p, model_name_test)

        mdl = RandomForestClassifier( random_state=42)
        mdl.fit(self.X_train0, self.y_train0)
            
        y_pred = mdl.predict_proba(self.X_train1)[:, 1]
        model_name_train1 = PATH + "/preds_train1/approach{}/rf_default.pkl.z".format(self.approach)
        jb.dump(y_pred, model_name_train1)
            
        p = mdl.predict_proba(self.valid)[:,1]
        model_name_val1 = PATH + "/preds_val1/approach{}/rf_default.pkl.z".format(self.approach)
        jb.dump(p, model_name_val1)

        precisions, recalls, thresholds = precision_recall_curve(self.y_valid, p)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (p >= thrs).astype(int)
        _, metric_rf = evaluate(y_pred, self.y_valid, plot_matrix=False)

        p = mdl.predict_proba(self.test)[:,1]
        model_name_test = PATH + "/preds_test/approach{}/rf_default.pkl.z".format(self.approach)
        jb.dump(p, model_name_test)

        print(f"Metrics -> lg = {metric_clf} | lgbm = {metric_lgbm} | xgb = {metric_xgb} | rf = {metric_rf}")

    def tune_nn_dropout(self, params):

        hidden1, hidden2, out1, out2, epoch, learning_rate = params
        model = keras.models.Sequential([
            keras.Input(shape=(self.X_train0.shape[1],), sparse=True, batch_size=32),
            keras.layers.Dense(hidden1, activation="relu"),
            keras.layers.Dropout(out1, seed=42),
            keras.layers.Dense(hidden2, activation="relu"),
            keras.layers.Dropout(out2, seed=42),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(loss=f1_loss,
                    optimizer=keras.optimizers.Adam(lr=learning_rate),
                    metrics=[f1_m,precision_m, recall_m])
            
        with tf.device(get_available_gpus()[0]):
            model.fit(self.X_train0, 
                        self.y_train0, 
                        epochs=epoch, 
                        validation_data=(self.valid, self.y_valid),
                        verbose=1)
        p = model.predict(self.X_train1)
        y_pred = (p >= 0.5).astype(int)

        model_name_train1 = PATH + "./preds_train1/approach{}/nn_dropout_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, hidden1, hidden2, out1, out2, epoch, learning_rate) 
        jb.dump(y_pred, model_name_train1)

        p = model.predict(self.valid)
        y_pred = (p >= 0.5).astype(int)
        _, metric = evaluate(y_pred, self.y_valid, plot_matrix=False)
        model_name_val1 = PATH + "/preds_val1/approach{}/nn_dropout_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, hidden1, hidden2, out1, out2, epoch, learning_rate) 
        jb.dump(p, model_name_val1)
            
        p = model.predict(self.test)
        y_pred = (p >= 0.5).astype(int)
        model_name_test = PATH + "./preds_test/approach{}/nn_dropout_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, hidden1, hidden2, out1, out2, epoch, learning_rate) 
        jb.dump(p, model_name_test)
        print(params, metric)
        print()
        return -metric

class ModelFeaturesGroups:
    def __init__(self, groups, approach, pipeline, X, train_size, y_train):

        self.groups = groups
        self.approach = approach
        self.pipeline = pipeline
        self.X = X.copy()
        self.train_size = train_size
        self.y_train = y_train

    def LGBM_1_group(self, params):

        for key, value in self.groups.items():
            X_ = self.X[value]
            X_prepared = self.pipeline.fit_transform(X_)
            X, test = X_prepared[:self.train_size, :], X_prepared[self.train_size:,:]
            train, valid, y_train, y_valid = train_test_split(X, self.y_train, test_size=0.4, random_state=42)
            X_train0, X_train1, y_train0, y_train1 = train_test_split(train, y_train, test_size=0.4, random_state=42)

            num_leaves, min_data_in_leaf, n_estimators, learning_rate = params
            mdl = LGBMClassifier(num_leaves=num_leaves,min_child_samples=min_data_in_leaf, learning_rate=learning_rate, 
                                n_estimators=n_estimators, random_state=42)
            mdl.fit(X_train0, y_train0)

            y_pred = mdl.predict_proba(X_train1)[:, 1]
            model_name_train1 = PATH + "/preds_train1/approach{}/lgbm_feature_groups_{}.pkl.z".format(self.approach, key) 
            jb.dump(y_pred, model_name_train1)
            
            p = mdl.predict_proba(valid)[:,1]
            model_name_val1 = PATH + "/preds_val1/approach{}/lgbm_feature_groups_{}.pkl.z".format(self.approach, key) 
            jb.dump(p, model_name_val1)

            precisions, recalls, thresholds = precision_recall_curve(y_valid, p)
            thrs, _ = better_threshold(precisions, recalls, thresholds)
            y_pred = (p >= thrs).astype(int)
            _, metric = evaluate(y_pred, y_valid, plot_matrix=False)
            
            p = mdl.predict_proba(test)[:,1]
            model_name_test = PATH + "/preds_test/approach{}/lgbm_feature_groups_{}.pkl.z".format(self.approach, key) 
            jb.dump(p, model_name_test)
            
            print(key, metric)
            print()

    def LGBM_2_group(self, params):

        for key1, key2 in combinations(self.groups.keys(), 2):

            selected = self.groups[key1] + self.groups[key2]
            X_ = self.X[selected]
            X_prepared = self.pipeline.fit_transform(X_)
            X, test = X_prepared[:self.train_size, :], X_prepared[self.train_size:,:]
            train, valid, y_train, y_valid = train_test_split(X, self.y_train, test_size=0.4, random_state=42)
            X_train0, X_train1, y_train0, y_train1 = train_test_split(train, y_train, test_size=0.4, random_state=42)
            num_leaves, min_data_in_leaf, n_estimators, learning_rate = params
            mdl = LGBMClassifier(num_leaves=num_leaves,min_child_samples=min_data_in_leaf, learning_rate=learning_rate, 
                                n_estimators=n_estimators, random_state=42)

            mdl.fit(X_train0, y_train0)

            y_pred = mdl.predict_proba(X_train1)[:, 1]
            model_name_train1 = PATH + "/preds_train1/approach{}/lgbm_feature_groups_{}_{}.pkl.z".format(self.approach, key1, key2) 
            jb.dump(y_pred, model_name_train1)
            
            p = mdl.predict_proba(valid)[:,1]
            model_name_val1 = PATH + "/preds_val1/approach{}/lgbm_feature_groups_{}_{}.pkl.z".format(self.approach, key1, key2) 
            jb.dump(p, model_name_val1)

            precisions, recalls, thresholds = precision_recall_curve(y_valid, p)
            thrs, _ = better_threshold(precisions, recalls, thresholds)
            y_pred = (p >= thrs).astype(int)
            _, metric = evaluate(y_pred, y_valid, plot_matrix=False)
            
            p = mdl.predict_proba(test)[:,1]
            model_name_test = PATH + "/preds_test/approach{}/lgbm_feature_groups_{}_{}.pkl.z".format(self.approach, key1, key2) 
            jb.dump(p, model_name_test)
            
            print(key1, key2, metric)
            print()


    def LR_1_group(self, params):

        for key, value in self.groups.items():
            print(value)
            X_ = self.X[value]
            X_prepared = self.pipeline.fit_transform(X_)
            X, test = X_prepared[:self.train_size, :], X_prepared[self.train_size:,:]
            train, valid, y_train, y_valid = train_test_split(X, self.y_train, test_size=0.4, random_state=42)
            X_train0, X_train1, y_train0, y_train1 = train_test_split(train, y_train, test_size=0.4, random_state=42)
            tol, max_iter, C = params
            mdl = LogisticRegression(random_state=42, 
                                solver='liblinear', 
                                max_iter=max_iter, 
                                tol=tol,
                                C=C,
                                penalty='l1')
            mdl.fit(X_train0, y_train0)

            y_pred = mdl.predict_proba(X_train1)[:, 1]
            model_name_train1 = PATH + "/preds_train1/approach{}/lr_feature_groups_{}.pkl.z".format(self.approach, key) 
            jb.dump(y_pred, model_name_train1)
            
            p = mdl.predict_proba(valid)[:,1]
            model_name_val1 = PATH + "/preds_val1/approach{}/lr_feature_groups_{}.pkl.z".format(self.approach, key) 
            jb.dump(p, model_name_val1)

            precisions, recalls, thresholds = precision_recall_curve(y_valid, p)
            thrs, _ = better_threshold(precisions, recalls, thresholds)
            y_pred = (p >= thrs).astype(int)
            _, metric = evaluate(y_pred, y_valid, plot_matrix=False)
            
            p = mdl.predict_proba(test)[:,1]
            model_name_test = PATH + "/preds_test/approach{}/lr_feature_groups_{}.pkl.z".format(self.approach, key) 
            jb.dump(p, model_name_test)
            
            print(key, metric)
            print()

    def LR_2_group(self, params):

        for key1, key2 in combinations(self.groups.keys(), 2):

            selected = self.groups[key1] + self.groups[key2]
            X_ = self.X[selected]
            X_prepared = self.pipeline.fit_transform(X_)
            X, test = X_prepared[:self.train_size, :], X_prepared[self.train_size:,:]
            train, valid, y_train, y_valid = train_test_split(X, self.y_train, test_size=0.4, random_state=42)
            X_train0, X_train1, y_train0, y_train1 = train_test_split(train, y_train, test_size=0.4, random_state=42)
            tol, max_iter, C = params
            mdl = LogisticRegression(random_state=42, 
                                solver='liblinear', 
                                max_iter=max_iter, 
                                tol=tol,
                                C=C,
                                penalty='l1')

            mdl.fit(X_train0, y_train0)

            y_pred = mdl.predict_proba(X_train1)[:, 1]
            model_name_train1 = PATH + "/preds_train1/approach{}/lr_feature_groups_{}_{}.pkl.z".format(self.approach, key1, key2) 
            jb.dump(y_pred, model_name_train1)
            
            p = mdl.predict_proba(valid)[:,1]
            model_name_val1 = PATH + "/preds_val1/approach{}/lr_feature_groups_{}_{}.pkl.z".format(self.approach, key1, key2) 
            jb.dump(p, model_name_val1)

            precisions, recalls, thresholds = precision_recall_curve(y_valid, p)
            thrs, _ = better_threshold(precisions, recalls, thresholds)
            y_pred = (p >= thrs).astype(int)
            _, metric = evaluate(y_pred, y_valid, plot_matrix=False)
            
            p = mdl.predict_proba(test)[:,1]
            model_name_test = PATH + "/preds_test/approach{}/lr_feature_groups_{}_{}.pkl.z".format(self.approach, key1, key2) 
            jb.dump(p, model_name_test)
            
            print(key1, key2, metric)
            print()

class ModelRowGroups:
    def __init__(self, approach, train0, train1, valid, test):
        self.approach = approach
        self.X_train0, self.y_train0 = train0
        self.X_train1, self.y_train1 = train1
        self.valid, self.y_valid = valid
        self.test = test
        self.make_buckets()

    def make_buckets(self):
        Xsum = pd.DataFrame(self.X_train0.sum(axis=1))
        n = int(max(Xsum.to_numpy())[0]//4)
        list_ns = [n, n*2]

        self.buckets_Xtr0 = np.digitize(pd.DataFrame(self.X_train0.sum(axis=1)).to_numpy(), bins=list_ns).reshape(1, -1)[0]
        self.buckets_Xtr1 = np.digitize(pd.DataFrame(self.X_train1.sum(axis=1)).to_numpy(), bins=list_ns).reshape(1, -1)[0]
        self.buckets_valid = np.digitize(pd.DataFrame(self.valid.sum(axis=1)).to_numpy(), bins=list_ns).reshape(1, -1)[0]
        self.buckets_test = np.digitize(pd.DataFrame(self.test.sum(axis=1)).to_numpy(), bins=list_ns).reshape(1, -1)[0]

    def XGboost(self):

        p_tr1 = np.zeros(self.X_train1.shape[0])
        p_vl1 = np.zeros(self.valid.shape[0])
        p_ts = np.zeros(self.test.shape[0])

        for bucket in np.unique(self.buckets_Xtr0):

            Xtr0 = self.X_train0[self.buckets_Xtr0 == bucket, :]
            ytr0 = self.y_train0[self.buckets_Xtr0 == bucket]


            mdl = XGBClassifier(n_jobs=-1, random_state=42)
            mdl.fit(Xtr0, ytr0)

            Xtr1 = self.X_train1[self.buckets_Xtr1 == bucket, :]
            p_tr1[self.buckets_Xtr1 == bucket] = mdl.predict_proba(Xtr1)[:, 1]

            Xval1 = self.valid[self.buckets_valid == bucket, :]
            p_vl1[self.buckets_valid == bucket] = mdl.predict_proba(Xval1)[:, 1]

            Xts = self.test[self.buckets_test == bucket, :]
            p_ts[self.buckets_test == bucket] = mdl.predict_proba(Xts)[:, 1]

        model_name_train1 = PATH + "/preds_train1/approach{}/xgb_rows_groups_buckets.z".format(self.approach)
        jb.dump(p_tr1, model_name_train1)
        model_name_val1 = PATH + "/preds_val1/approach{}/xgb_rows_groups_buckets.pkl.z".format(self.approach)
        jb.dump(p_vl1, model_name_val1)
        model_name_test = PATH + "/preds_test/approach{}/xgb_rows_groups_buckets.pkl.z".format(self.approach)
        jb.dump(p_ts, model_name_test)

        precisions, recalls, thresholds = precision_recall_curve(self.y_valid, p_vl1)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (p_vl1 >= thrs).astype(int)
        _, metrics = evaluate(y_pred, self.y_valid, plot_matrix=False)

        print(metrics)
        print()

    def LGBM(self):

        p_tr1 = np.zeros(self.X_train1.shape[0])
        p_vl1 = np.zeros(self.valid.shape[0])
        p_ts = np.zeros(self.test.shape[0])

        for bucket in np.unique(self.buckets_Xtr0):

            Xtr0 = self.X_train0[self.buckets_Xtr0 == bucket, :]
            ytr0 = self.y_train0[self.buckets_Xtr0 == bucket]


            mdl = LogisticRegression(random_state=42)
            mdl.fit(Xtr0, ytr0)

            Xtr1 = self.X_train1[self.buckets_Xtr1 == bucket, :]
            p_tr1[self.buckets_Xtr1 == bucket] = mdl.predict_proba(Xtr1)[:, 1]

            Xval1 = self.valid[self.buckets_valid == bucket, :]
            p_vl1[self.buckets_valid == bucket] = mdl.predict_proba(Xval1)[:, 1]

            Xts = self.test[self.buckets_test == bucket, :]
            p_ts[self.buckets_test == bucket] = mdl.predict_proba(Xts)[:, 1]

        model_name_train1 = PATH + "/preds_train1/approach{}/xgb_rows_groups_buckets.z".format(self.approach)
        jb.dump(p_tr1, model_name_train1)
        model_name_val1 = PATH + "/preds_val1/approach{}/xgb_rows_groups_buckets.pkl.z".format(self.approach)
        jb.dump(p_vl1, model_name_val1)
        model_name_test = PATH + "/preds_test/approach{}/xgb_rows_groups_buckets.pkl.z".format(self.approach)
        jb.dump(p_ts, model_name_test)

        precisions, recalls, thresholds = precision_recall_curve(self.y_valid, p_vl1)
        thrs, _ = better_threshold(precisions, recalls, thresholds)
        y_pred = (p_vl1 >= thrs).astype(int)
        _, metrics = evaluate(y_pred, self.y_valid, plot_matrix=False)

        print(metrics)
        print()