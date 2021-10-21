from scipy import sparse
from tqdm import tqdm_notebook
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils import *
# Tune Models
# ===================================================================

class TuneModels:
    def __init__(self, PATH, approach, train0, train1, valid, test, n_folders=5):
        self.path = PATH
        self.approach = approach
        self.X_train0 = train0
        self.X_train1 = train1
        self.valid = valid
        self.test = test
        self.n_folders = n_folders


    def return_data(self):
        X_train0 = self.X_train0.tocsr()[:, :-2]
        X_train1 = self.X_train1.tocsr()[:, :-2]
        X_valid = self.valid.tocsr()[:, :-2]
        y_train0 = self.X_train0.tocsr()[:, -2].A.reshape(1, -1)[0]
        y_train1 = self.X_train1.tocsr()[:, -2].A.reshape(1, -1)[0]
        y_valid = self.valid.tocsr()[:, -2].A.reshape(1, -1)[0]

        return X_train0, X_train1, X_valid, y_train0, y_train1, y_valid
    
    def Model(self, model_name, params=None, features_size=None):
        if model_name == "NN":
            hidden1, hidden2, _, learning_rate = params
            model = keras.models.Sequential([
                keras.Input(shape=features_size, sparse=True),
                keras.layers.Dense(hidden1, activation="relu"),
                keras.layers.Dense(hidden2, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid")
            ])
            model.compile(loss=f1_loss,
                        optimizer=keras.optimizers.Adam(lr=learning_rate),
                        metrics=[f1_m,precision_m, recall_m])
            
            return model
        
        if model_name == "logistic":
            tol, max_iter, C = params
            model = LogisticRegression(random_state=42, 
                            solver='liblinear', 
                            max_iter=max_iter, 
                            tol=tol,
                            C=C,
                            penalty='l1')

            return model

        if model_name == "lgbm":
            num_leaves, min_data_in_leaf, n_estimators, learning_rate = params
            model = LGBMClassifier(num_leaves=num_leaves,
                                    min_child_samples=min_data_in_leaf, 
                                    learning_rate=learning_rate, 
                                    n_estimators=n_estimators, 
                                    random_state=42)

            return model

        if model_name == "xgboost":
            learning_rate = params[0]
            max_depth = params[1]
            min_child_weight=params[2]
            subsample = params[3]
            colsample_bynode = params[4]
            num_parallel_tree = params[5]
            n_estimators = params[6]
            if len(get_available_gpus()) > 0:
                model = XGBClassifier(
                    n_jobs=-1,
                    eval_metric='auc',
                    random_state=42,
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bynode=colsample_bynode,
                    num_parallel_tree=num_parallel_tree,
                    tree_method='gpu_hist', 
                    gpu_id=0
                )
            else:
                model = XGBClassifier(
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

            return model

        if model_name == "rf":
            min_samples_leaf, weight, max_depth, n_estimators = params
            model = RandomForestClassifier(n_estimators=n_estimators, 
                                            min_samples_leaf=min_samples_leaf, 
                                            max_depth = max_depth,
                                            class_weight = {0:weight, 1: 1},
                                            random_state=42)
            
            return model

        if model_name == "knn":
            n_neighbors, leaf_size, p, weights, metric = params
            model = KNeighborsClassifier(n_neighbors=n_neighbors, 
                                            leaf_size=leaf_size,
                                            p=p,
                                            weights=weights,
                                            metric=metric,
                                            n_jobs=-1)
            return model

        if model_name == "default":
            clf = LogisticRegression(random_state=42)
            lgbm = LGBMClassifier(random_state=42)
            xgb = XGBClassifier(n_jobs=-1, random_state=42)
            rf = RandomForestClassifier( random_state=42)

            return clf, lgbm, xgb, rf

        if model_name == "NN_dropout":
            hidden1, hidden2, out1, out2, _, learning_rate = params
            model = keras.models.Sequential([
                keras.Input(shape=features_size, sparse=True),
                keras.layers.Dense(hidden1, activation="de"),
                keras.layers.Dropout(out1),
                keras.layers.Dense(hidden2, activation="relu"),
                keras.layers.Dropout(out2),
                keras.layers.Dense(1, activation="sigmoid")
            ])
            model.compile(loss=f1_loss,
                        optimizer=keras.optimizers.Adam(lr=learning_rate),
                        metrics=[f1_m,precision_m, recall_m])
            
            return model

    def tune_nn(self, params):

        hidden1, hidden2, epoch, learning_rate = params
        X = sparse.vstack((self.X_train0, self.X_train1, self.valid))
        scores = []
        for fold in tqdm_notebook(range(self.n_folders), desc="Cross validation progress"):
            train_data = X.tocsr()[(X.tocsr()[:, -1].A != fold).nonzero()[0], :].copy()
            valid_data = X.tocsr()[(X.tocsr()[:, -1].A == fold).nonzero()[0], :].copy()
            X_train = train_data.tocsr()[:, :-2]
            X_valid = valid_data.tocsr()[:, :-2]
            y_train = train_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            y_valid = valid_data.tocsr()[:, -2].A.reshape(1, -1)[0]

            model = self.Model("NN", params, X_train.shape[1])
                
            if len(get_available_gpus()) > 0:
                with tf.device(get_available_gpus()[0]):
                    model.fit(X_train, 
                                y_train, 
                                epochs=epoch, 
                                validation_data=(X_valid, y_valid),
                                verbose=0)
            else:
                model.fit(X_train, 
                                y_train, 
                                epochs=epoch, 
                                validation_data=(X_valid, y_valid),
                                verbose=0)

            p = model.predict(X_train)
            thrs, _ = better_threshold(y_train, p)
            y_pred = model.predict(X_valid)
            y_pred = (y_pred.reshape(1,-1)[0] >= thrs).astype(int)
            scores.append(evaluate(y_valid, y_pred)[1])


        X_train0, X_train1, X_valid, y_train0, _, y_valid = self.return_data()        
        
        model = self.Model("NN", params, X_train0.shape[1])
            
        if len(get_available_gpus()) > 0:
            with tf.device(get_available_gpus()[0]):
                model.fit(X_train0, 
                            y_train0, 
                            epochs=epoch, 
                            validation_data=(X_valid, y_valid),
                            verbose=0)
        else:
            model.fit(X_train0, 
                            y_train0, 
                            epochs=epoch, 
                            validation_data=(X_valid, y_valid),
                            verbose=0)
        p = model.predict(X_train1)

        model_name_train1 = "/preds_train1/approach{}/nn_{}_{}_{}_{}.pkl.z".format(self.approach,hidden1, hidden2, epoch, learning_rate) 
        jb.dump(p, self.path + model_name_train1)

        p = model.predict(X_valid)
        model_name_val1 = "/preds_val1/approach{}/nn_{}_{}_{}_{}.pkl.z".format(self.approach,hidden1, hidden2, epoch, learning_rate) 
        jb.dump(p, self.path + model_name_val1)
            
        p = model.predict(self.test)
        model_name_test = "/preds_test/approach{}/nn_{}_{}_{}_{}.pkl.z".format(self.approach,hidden1, hidden2, epoch, learning_rate) 
        jb.dump(p, self.path + model_name_test)


        metric = np.mean(scores)    
        print(params, metric)
        print()
        return -metric

    def tune_logistic(self, params):
        tol, max_iter, C = params
        X = sparse.vstack((self.X_train0, self.X_train1, self.valid))
        scores = []
        for fold in tqdm_notebook(range(self.n_folders), desc="Cross validation progress"):
            train_data = X.tocsr()[(X.tocsr()[:, -1].A != fold).nonzero()[0], :].copy()
            valid_data = X.tocsr()[(X.tocsr()[:, -1].A == fold).nonzero()[0], :].copy()
            X_train = train_data.tocsr()[:, :-2]
            X_valid = valid_data.tocsr()[:, :-2]
            y_train = train_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            y_valid = valid_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            
            clf = self.Model("logistic", params)
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_train)[:, 1]
            thrs, _ = better_threshold(y_train, y_pred)
            y_pred = clf.predict_proba(X_valid)[:, 1]
            y_pred = (y_pred >= thrs).astype(int)
            scores.append(evaluate(y_valid, y_pred)[1])
            
        X_train0, X_train1, X_valid, y_train0, _ , y_valid = self.return_data()   
        clf = self.Model("logistic", params)

        clf.fit(X_train0, y_train0)

        y_pred = clf.predict_proba(X_train1)[:, 1]
        model_name_train1 ="/preds_train1/approach{}/lr_{}_{}_{}.pkl.z".format(self.approach, tol, max_iter, C) 
        jb.dump(y_pred, self.path + model_name_train1)
        
        y_pred = clf.predict_proba(X_valid)[:, 1]
        model_name_val1 = "/preds_val1/approach{}/lr_{}_{}_{}.pkl.z".format(self.approach, tol, max_iter, C) 
        jb.dump(y_pred, self.path + model_name_val1)
        
        y_pred = clf.predict_proba(self.test)[:, 1]
        model_name_test = "/preds_test/approach{}/lr_{}_{}_{}.pkl.z".format(self.approach, tol, max_iter, C) 
        jb.dump(y_pred, self.path + model_name_test)
    

        metric = np.mean(scores)
        print(params, metric)
        print()
        
        return -metric

    def tune_lgbm(self, params):
        num_leaves, min_data_in_leaf, n_estimators, learning_rate = params
        X = sparse.vstack((self.X_train0, self.X_train1, self.valid))
        scores = []
        for fold in tqdm_notebook(range(self.n_folders), desc="Cross validation progress"):
            train_data = X.tocsr()[(X.tocsr()[:, -1].A != fold).nonzero()[0], :].copy()
            valid_data = X.tocsr()[(X.tocsr()[:, -1].A == fold).nonzero()[0], :].copy()
            X_train = train_data.tocsr()[:, :-2]
            X_valid = valid_data.tocsr()[:, :-2]
            y_train = train_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            y_valid = valid_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            
            mdl = self.Model("lgbm", params)
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict_proba(X_train)[:, 1]
            thrs, _ = better_threshold(y_train, y_pred)
            y_pred = mdl.predict_proba(X_valid)[:, 1]
            y_pred = (y_pred >= thrs).astype(int)
            scores.append(evaluate(y_valid, y_pred)[1])

        
        X_train0, X_train1, X_valid, y_train0, _ , y_valid = self.return_data()  
        mdl = self.Model("lgbm", params)
        mdl.fit(X_train0, y_train0)

        y_pred = mdl.predict_proba(X_train1)[:, 1]
        model_name_train1 = "/preds_train1/approach{}/lgbm_{}_{}_{}_{}.pkl.z".format(self.approach, num_leaves, min_data_in_leaf, n_estimators, learning_rate) 
        jb.dump(y_pred, self.path + model_name_train1)
        
        p = mdl.predict_proba(X_valid)[:,1]
        model_name_val1 = "/preds_val1/approach{}/lgbm_{}_{}_{}_{}.pkl.z".format(self.approach, num_leaves, min_data_in_leaf, n_estimators, learning_rate) 
        jb.dump(p, self.path + model_name_val1)
        
        p = mdl.predict_proba(self.test)[:,1]
        model_name_test = "/preds_test/approach{}/lgbm_{}_{}_{}_{}.pkl.z".format(self.approach, num_leaves, min_data_in_leaf, n_estimators, learning_rate) 
        jb.dump(p, self.path + model_name_test)
        
        metric = np.mean(scores)
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
    
        X = sparse.vstack((self.X_train0, self.X_train1, self.valid))
        scores = []
        for fold in tqdm_notebook(range(self.n_folders), desc="Cross validation progress"):
            train_data = X.tocsr()[(X.tocsr()[:, -1].A != fold).nonzero()[0], :].copy()
            valid_data = X.tocsr()[(X.tocsr()[:, -1].A == fold).nonzero()[0], :].copy()
            X_train = train_data.tocsr()[:, :-2]
            X_valid = valid_data.tocsr()[:, :-2]
            y_train = train_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            y_valid = valid_data.tocsr()[:, -2].A.reshape(1, -1)[0]

            fit_params = {
                'early_stopping_rounds': 100,
                'eval_metric' : 'auc',
                'eval_set': [(X_valid, y_valid)],
                'verbose': False,
            }
            xgb = self.Model("xgboost", params)
            xgb.fit(X_train, y_train, **fit_params)
            y_pred = xgb.predict_proba(X_train)[:, 1]
            thrs, _ = better_threshold(y_train, y_pred)
            y_pred = xgb.predict_proba(X_valid)[:, 1]
            y_pred = (y_pred >= thrs).astype(int)
            scores.append(evaluate(y_valid, y_pred)[1])
        
        X_train0, X_train1, X_valid, y_train0, _ , y_valid = self.return_data()  
        xgb = self.Model("xgboost", params)
        fit_params = {
                'early_stopping_rounds': 100,
                'eval_metric' : 'auc',
                'eval_set': [(X_valid, y_valid)],
                'verbose': False,
            }
        xgb.fit(X_train0, y_train0, **fit_params)

        y_pred = xgb.predict_proba(X_train1)[:,1]
        model_name_train1 = "/preds_train1/approach{}/xgb_{}_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, learning_rate, max_depth, min_child_weight, subsample, colsample_bynode, num_parallel_tree, n_estimators) 
        jb.dump(y_pred, self.path + model_name_train1)
        
        p = xgb.predict_proba(X_valid)[:,1]
        model_name_val1 = "/preds_val1/approach{}/xgb_{}_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, learning_rate, max_depth, min_child_weight, subsample, colsample_bynode, num_parallel_tree, n_estimators)
        jb.dump(p, self.path + model_name_val1)
        
        p = xgb.predict_proba(self.test)[:,1]
        model_name_test = "/preds_test/approach{}/xgb_{}_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, learning_rate, max_depth, min_child_weight, subsample, colsample_bynode, num_parallel_tree, n_estimators)
        jb.dump(p, self.path + model_name_test)
        
        metric = np.mean(scores)
        print(params, metric)
        print()
        
        return -metric

    def tune_trees(self, params):
        min_samples_leaf, weight, max_depth, n_estimators = params
        scores = []
        X = sparse.vstack((self.X_train0, self.X_train1, self.valid))
        scores = []
        for fold in tqdm_notebook(range(self.n_folders), desc="Cross validation progress"):
            train_data = X.tocsr()[(X.tocsr()[:, -1].A != fold).nonzero()[0], :].copy()
            valid_data = X.tocsr()[(X.tocsr()[:, -1].A == fold).nonzero()[0], :].copy()
            X_train = train_data.tocsr()[:, :-2]
            X_valid = valid_data.tocsr()[:, :-2]
            y_train = train_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            y_valid = valid_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            rf = self.Model("rf", params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict_proba(X_train)[:, 1]
            thrs, _ = better_threshold(y_train, y_pred)
            y_pred = rf.predict_proba(X_valid)[:, 1]
            y_pred = (y_pred >= thrs).astype(int)
            scores.append(evaluate(y_valid, y_pred)[1])

        X_train0, X_train1, X_valid, y_train0, _ , y_valid = self.return_data() 
        rf = self.Model("rf", params)
        rf.fit(X_train0, y_train0)

        y_pred = rf.predict_proba(X_train1)[:,1]
        model_name_train1 = "/preds_train1/approach{}/rf_{}_{}_{}_{}.pkl.z".format(self.approach, min_samples_leaf, weight, max_depth, n_estimators) 
        jb.dump(y_pred, self.path + model_name_train1)
        
        p = rf.predict_proba(X_valid)[:, 1]
        model_name_val1 = "/preds_val1/approach{}/rf_{}_{}_{}_{}.pkl.z".format(self.approach, min_samples_leaf, weight, max_depth, n_estimators) 
        jb.dump(p, self.path + model_name_val1)
        
        p = rf.predict_proba(self.test)[:,1]
        model_name_test = "/preds_test/approach{}/rf_{}_{}_{}_{}.pkl.z".format(self.approach, min_samples_leaf, weight, max_depth, n_estimators) 
        jb.dump(p, self.path + model_name_test)
        
        metric = np.mean(scores)
        print(params, metric)
        print()
        
        return -metric

    def tune_knn(self, params):
        n_neighbors, leaf_size, p, wheights, metric = params
        X = sparse.vstack((self.X_train0, self.X_train1, self.valid))
        scores = []
        for fold in tqdm_notebook(range(self.n_folders), desc="Cross validation progress"):
            train_data = X.tocsr()[(X.tocsr()[:, -1].A != fold).nonzero()[0], :].copy()
            valid_data = X.tocsr()[(X.tocsr()[:, -1].A == fold).nonzero()[0], :].copy()
            X_train = train_data.tocsr()[:, :-2]
            X_valid = valid_data.tocsr()[:, :-2]
            y_train = train_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            y_valid = valid_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            neigh = self.Model("knn", params)
            neigh.fit(X_train, y_train)
            y_pred = neigh.predict_proba(X_train)[:, 1]
            thrs, _ = better_threshold(y_train, y_pred)
            y_pred = neigh.predict_proba(X_valid)[:, 1]
            y_pred = (y_pred >= thrs).astype(int)
            scores.append(evaluate(y_valid, y_pred)[1])

        X_train0, X_train1, X_valid, y_train0, _ , y_valid = self.return_data() 
        neigh = self.Model("knn", params)
        neigh.fit(X_train0, y_train0)

        y_pred = neigh.predict_proba(X_train1)[:,1]
        model_name_train1 = "/preds_train1/approach{}/knn_{}_{}_{}_{}_{}.pkl.z".format(self.approach, n_neighbors, leaf_size, p, wheights, metric)
        jb.dump(y_pred, self.path + model_name_train1)
        
        y_pred = neigh.predict_proba(X_valid)[:, 1]
        model_name_val1 = "/preds_val1/approach{}/knn_{}_{}_{}_{}_{}.pkl.z".format(self.approach, n_neighbors, leaf_size, p, wheights, metric)
        jb.dump(y_pred, self.path + model_name_val1)
        
        y_pred = neigh.predict_proba(self.test)[:,1]
        model_name_test = "/preds_test/approach{}/knn_{}_{}_{}_{}_{}.pkl.z".format(self.approach, n_neighbors, leaf_size, p, wheights, metric)
        jb.dump(y_pred, self.path + model_name_test)
        
        metric = np.mean(scores)
        print(params, metric)
        print()
        
        return -metric

    def default_models(self):
        scores_clf, scores_lgbm, scores_xgb, scores_rf = [], [], [], []
        X = sparse.vstack((self.X_train0, self.X_train1, self.valid))
        for fold in tqdm_notebook(range(self.n_folders), desc="Cross validation progress"):
            train_data = X.tocsr()[(X.tocsr()[:, -1].A != fold).nonzero()[0], :].copy()
            valid_data = X.tocsr()[(X.tocsr()[:, -1].A == fold).nonzero()[0], :].copy()
            X_train = train_data.tocsr()[:, :-2]
            X_valid = valid_data.tocsr()[:, :-2]
            y_train = train_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            y_valid = valid_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            clf, lgbm, xgb, rf = self.Model("default")
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_train)[:, 1]
            thrs, _ = better_threshold(y_train, y_pred)
            y_pred = clf.predict_proba(X_valid)[:, 1]
            y_pred = (y_pred >= thrs).astype(int)
            scores_clf.append(evaluate(y_valid, y_pred)[1])
            lgbm.fit(X_train, y_train)
            y_pred = lgbm.predict_proba(X_train)[:, 1]
            thrs, _ = better_threshold(y_train, y_pred)
            y_pred = lgbm.predict_proba(X_valid)[:, 1]
            y_pred = (y_pred >= thrs).astype(int)
            scores_lgbm.append(evaluate(y_valid, y_pred)[1])
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict_proba(X_train)[:, 1]
            thrs, _ = better_threshold(y_train, y_pred)
            y_pred = xgb.predict_proba(X_valid)[:, 1]
            y_pred = (y_pred >= thrs).astype(int)
            scores_xgb.append(evaluate(y_valid, y_pred)[1])
            rf.fit(X_train, y_train)
            y_pred = rf.predict_proba(X_train)[:, 1]
            thrs, _ = better_threshold(y_train, y_pred)
            y_pred = rf.predict_proba(X_valid)[:, 1]
            y_pred = (y_pred >= thrs).astype(int)
            scores_rf.append(evaluate(y_valid, y_pred)[1])
        

        X_train0, X_train1, X_valid, y_train0, _ , y_valid = self.return_data() 
        clf, lgbm, xgb, rf = self.Model("default")
        clf.fit(X_train0,y_train0)

        y_pred = clf.predict_proba(X_train1)[:, 1]
        model_name_train1 = "/preds_train1/approach{}/lr_default.pkl.z".format(self.approach) 
        jb.dump(y_pred, self.path + model_name_train1)
        
        y_pred = clf.predict_proba(X_valid)[:, 1]
        model_name_val1 = "/preds_val1/approach{}/lr_default.pkl.z".format(self.approach) 
        jb.dump(y_pred, self.path + model_name_val1)
        
        y_pred = clf.predict_proba(self.test)[:, 1]
        model_name_test = "/preds_test/approach{}/lr_default.pkl.z".format(self.approach) 
        jb.dump(y_pred, self.path + model_name_test)

        lgbm.fit(X_train0, y_train0)

        y_pred = lgbm.predict_proba(X_train1)[:, 1]
        model_name_train1 = "/preds_train1/approach{}/lgbm_default.pkl.z".format(self.approach)
        jb.dump(y_pred, self.path + model_name_train1)
            
        p = lgbm.predict_proba(X_valid)[:,1]
        model_name_val1 = "/preds_val1/approach{}/lgbm_default.pkl.z".format(self.approach)
        jb.dump(p, self.path + model_name_val1)

        p = lgbm.predict_proba(self.test)[:,1]
        model_name_test = "/preds_test/approach{}/lgbm_default.pkl.z".format(self.approach)
        jb.dump(p, self.path + model_name_test)


        xgb.fit(X_train0, y_train0)

        y_pred = xgb.predict_proba(X_train1)[:, 1]
        model_name_train1 = "/preds_train1/approach{}/xgboost_default.pkl.z".format(self.approach)
        jb.dump(y_pred, self.path + model_name_train1)
            
        p = xgb.predict_proba(X_valid)[:,1]
        model_name_val1 = "/preds_val1/approach{}/xgboost_default.pkl.z".format(self.approach)
        jb.dump(p, self.path + model_name_val1)

        p = xgb.predict_proba(self.test)[:,1]
        model_name_test = "/preds_test/approach{}/xgboost_default.pkl.z".format(self.approach)
        jb.dump(p, self.path + model_name_test)

        rf.fit(X_train0, y_train0)

        y_pred = rf.predict_proba(X_train1)[:, 1]
        model_name_train1 = "/preds_train1/approach{}/rf_default.pkl.z".format(self.approach)
        jb.dump(y_pred, self.path + model_name_train1)
            
        p = rf.predict_proba(X_valid)[:,1]
        model_name_val1 = "/preds_val1/approach{}/rf_default.pkl.z".format(self.approach)
        jb.dump(p, self.path + model_name_val1)

        p = rf.predict_proba(self.test)[:,1]
        model_name_test = "/preds_test/approach{}/rf_default.pkl.z".format(self.approach)
        jb.dump(p, self.path + model_name_test)

        metric_clf = np.mean(scores_clf)
        metric_lgbm = np.mean(scores_lgbm)
        metric_xgb = np.mean(scores_xgb)
        metric_rf = np.mean(scores_rf)
        print(f"Metrics -> lg = {metric_clf} | lgbm = {metric_lgbm} | xgb = {metric_xgb} | rf = {metric_rf}")

    def tune_nn_dropout(self, params):
        hidden1, hidden2, out1, out2, epoch, learning_rate = params
        X = sparse.vstack((self.X_train0, self.X_train1, self.valid))
        scores = []
        for fold in tqdm_notebook(range(self.n_folders), desc="Cross validation progress"):
            train_data = X.tocsr()[(X.tocsr()[:, -1].A != fold).nonzero()[0], :].copy()
            valid_data = X.tocsr()[(X.tocsr()[:, -1].A == fold).nonzero()[0], :].copy()
            X_train = train_data.tocsr()[:, :-2]
            X_valid = valid_data.tocsr()[:, :-2]
            y_train = train_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            y_valid = valid_data.tocsr()[:, -2].A.reshape(1, -1)[0]
            model = self.Model("NN_dropout", params, X_train.shape[1])
            if len(get_available_gpus()) > 0:
                with tf.device(get_available_gpus()[0]):
                    model.fit(X_train, 
                                y_train, 
                                epochs=epoch, 
                                validation_data=(X_valid, y_valid),
                                verbose=0)
            else:
                model.fit(X_train, 
                            y_train, 
                            epochs=epoch, 
                            validation_data=(X_valid, y_valid),
                            verbose=0)

            p = model.predict(X_train)
            thrs, _ = better_threshold(y_train, p)
            y_pred = model.predict(X_valid)
            y_pred = (y_pred.reshape(1,-1)[0] >= thrs).astype(int)
            scores.append(evaluate(y_valid, y_pred)[1])

        X_train0, X_train1, X_valid, y_train0, _, y_valid = self.return_data()    
        model = self.Model("NN_dropout", params, X_train.shape[1])
            
        if len(get_available_gpus()) > 0:
                with tf.device(get_available_gpus()[0]):
                    model.fit(X_train0, 
                                y_train0, 
                                epochs=epoch, 
                                validation_data=(X_valid, y_valid),
                                verbose=0)
        else:
            model.fit(X_train0, 
                        y_train0, 
                        epochs=epoch, 
                        validation_data=(X_valid, y_valid),
                        verbose=0)
        y_pred = model.predict(X_train1)

        model_name_train1 = "/preds_train1/approach{}/nn_dropout_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, hidden1, hidden2, out1, out2, epoch, learning_rate) 
        jb.dump(y_pred, self.path + model_name_train1)

        y_pred = model.predict(X_valid)
        model_name_val1 = "/preds_val1/approach{}/nn_dropout_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, hidden1, hidden2, out1, out2, epoch, learning_rate) 
        jb.dump(y_pred, self.path + model_name_val1)
            
        y_pred = model.predict(self.test)
        model_name_test = "/preds_test/approach{}/nn_dropout_{}_{}_{}_{}_{}_{}.pkl.z".format(self.approach, hidden1, hidden2, out1, out2, epoch, learning_rate) 
        jb.dump(y_pred, self.path + model_name_test)

        metric = np.mean(scores)
        print(params, metric)
        print()
        return -metric
