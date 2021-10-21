import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans  
from skopt import gp_minimize

class ProcessMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, 
                columns,
                cat_columns,
                type_columns='categorical',
                num_imputer=None):
        self.columns = columns
        self.cat_columns = cat_columns
        self.type_columns = type_columns
        self.num_imputer = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.type_columns == 'categorical':
            X = self.transform_categorical(X)

        else:
            X = self.transform_numerical(X)

        return X

    def transform_categorical(self, X):
        for col in self.columns:
            X[col] = X[col].replace(-999, np.nan)
            X['NA_' + col] = X[col].isna().astype(np.int8)
            X[col].fillna('UNKNOWN', inplace=True)

        return X

    def transform_numerical(self, X):
        self.columns = [col for col in X.columns if 'NA_' not in col and col not in self.cat_columns]
        for col in self.columns:
            X[col] = X[col].replace(-999, np.nan)
            if self.num_imputer == None:
                imputer = X[col].dropna().median()
            else:
                imputer = self.imputer
            X['NA_' + col] = X[col].isna().astype(np.int8)
            X = X.fillna(imputer)

        return X

class Categorify(BaseEstimator, TransformerMixin):
    def __init__(self, 
                columns,
                freq_treshhold=5,
                lowfrequency_id=0,
                unkown_id=1):

        self.columns = columns
        self.freqs = []
        self.freq_treshhold = freq_treshhold
        self.lowfrequency_id = lowfrequency_id
        self.unkown_id = unkown_id

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.make_columns(X)
        for idx, col in enumerate(self.columns):
            col_name = self.freqs[idx].columns[0]
            X = X.merge(self.freqs[idx], how='left', on=col, suffixes=("_", ""))
            if col_name + "_" in X.columns:
                X.drop(col_name + "_", axis=1, inplace=True)


        return X

    def make_columns(self, X):
        self.freqs = []
        for col in self.columns:
            freq = X[col].value_counts()
            freq = freq.reset_index()
            freq.columns = [col, 'count']
            freq = freq.reset_index()
            freq.columns = [col + '_Categorify', col, 'count']
            freq[col + '_Categorify'] = freq[col + '_Categorify'] + 2
            freq.loc[freq['count'] < self.freq_treshhold, col + '_Categorify'] = self.lowfrequency_id
            freq.loc[freq[col]=='UNKNOWN', col + '_Categorify'] = self.unkown_id
            freq = freq.drop('count', axis=1)
            self.freqs.append(freq)

class GetRidCategoricalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            categority_col = col + '_Categorify'
            if col in X.columns:
                X.drop([col], axis=1,inplace=True)

        return X

class CountEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, 
                columns):
        self.columns = columns
        self.ces = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for idx, col in enumerate(self.columns):
            col_name = col + '_Categorify'
            ce = X[col_name].value_counts()
            ce = ce.reset_index()
            ce.columns = [col_name, 'CE_' + col]
            X = X.merge(ce, how='left', on=col_name)

        return X

class StandardNumerical(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.columns = [col for col in X.columns if "NA" not in col]
        for col in self.columns:
            X[col] = X[col].replace(-999, np.nan)
            median = X[col].dropna().median()
            if median != median:
                X.drop(col, axis=1, inplace=True)
            else:
                X[col].fillna(median, inplace=True)
                X[col]= (X[col] - np.mean(X[col]))/np.std(X[col])


        return X

class CombiningColumns(BaseEstimator, TransformerMixin):
    def __init__(self, list_combined, train_size, col_type='categorical'):
        self.list_combined = list_combined
        self.col_type = col_type
        self.space = [(2, 20),
                    (200, 1000),
                    (2, 20),
                    (1e-5, 1e-3, 'log-uniform')]
        self.train_size = train_size
        self.clusters = {}
        self.names_col = []

    def fit(self, X, y):
        if self.col_type == 'numerical':
            X_ = X[:self.train_size, :]
            y_ = y[:self.train_size]
            for cols in self.list_combined:
                def tune_kmeans(params):
                    n_clusters, max_iter, n_init, tol = params

                    kmeans = KMeans(n_clusters=n_clusters, 
                                    max_iter=max_iter, 
                                    tol=tol, n_init=n_init, 
                                    n_jobs=-1, 
                                    random_state=42)
                    cluster = df_copy.copy()
                    cluster["Cluster"] = kmeans.fit_predict(cluster, y)
                    cluster["Cluster"] = cluster["Cluster"].astype("category")
                    cluster["y"] = y_.to_numpy()
                    return -cluster[["Cluster", "y"]].groupby("Cluster").mean().var()["y"]

                df_copy = X_[cols].copy()
                res = gp_minimize(tune_kmeans, self.space, random_state=42, verbose=0, n_calls=30)

                name_col = cols[0] + "_" + cols[1]

                n_clusters, max_iter, n_init, tol = res.x
                kmeans = KMeans(n_clusters=n_clusters, 
                                max_iter=max_iter, 
                                tol=tol, n_init=n_init, 
                                n_jobs=-1, 
                                random_state=42)
                

                df_copy[name_col] = kmeans.fit_predict(df_copy, y)
                df_copy[name_col] = df_copy[name_col].astype("category")
                self.clusters[name_col] = kmeans
                self.names_col.append(name_col)

        return self

    def transform(self, X):
        if self.col_type == 'categorical':
            X = self.transform_cat(X)

        else:
            X = self.transform_num(X)

        return X

    def transform_cat(self, X):
        for cols in self.list_combined:
            if len(cols) == 2:
                name_col = cols[0] + '_' + cols[1]
                X[name_col] = X[cols[0]].astype(str) + "_" +  X[cols[1]].astype(str)
            else:
                name_col = cols[0] + '_' + cols[1] + '_' + cols[2]
                X[name_col] = X[cols[0]].astype(str) + "_" +  X[cols[1]].astype(str) + '_' + X[cols[2]].astype(str)

        return X

    def transform_num(self, X):
        for idx, cols in enumerate(self.list_combined):
            name_col = self.names_col[idx]
            X[name_col] = self.clusters[name_col].predict(X[cols].copy())
            X[name_col] = X[name_col].astype("category")


        return X