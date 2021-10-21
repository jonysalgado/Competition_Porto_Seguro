# =========================================================================
# Importations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def remove_feature(list_remove, columns):
    return [x for x in columns if x not in list_remove]


class ProcessMissingValues1(BaseEstimator, TransformerMixin):
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

class ProcessMissingValues2(BaseEstimator, TransformerMixin):
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
            # median = X[col].dropna().median()
            X['NA_' + col] = X[col].isna().astype(np.int8)
            X[col].fillna(-9999, inplace=True)
 
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

class Categorify1(BaseEstimator, TransformerMixin):
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
class Categorify2(BaseEstimator, TransformerMixin):
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
            df_copy = X.copy()
            df_copy[col] = df_copy[col].astype(str)
            df_copy = df_copy.merge(self.freqs[idx], how='left', on=col)
          
            X[col] = np.zeros(X.shape[0])
            X[col] = X[col].astype(int)
            X[col] = df_copy[col_name].copy()
            
 
 
 
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
            freq.loc[freq[col]==-9999, col + '_Categorify'] = self.unkown_id
            freq = freq.drop('count', axis=1)
            freq[col] = freq[col].astype(str)
            self.freqs.append(freq)

class GetRidCategoricalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            if col in X.columns:
                X.drop([col], axis=1,inplace=True)

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
    def __init__(self, list_combined):
        self.list_combined = list_combined

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for cols in self.list_combined:
            if len(cols) == 2:
                name_col = cols[0] + '_' + cols[1]
                X[name_col] = X[cols[0]].astype(str) + "_" +  X[cols[1]].astype(str)
            else:
                name_col = cols[0] + '_' + cols[1] + '_' + cols[2]
                X[name_col] = X[cols[0]].astype(str) + "_" +  X[cols[1]].astype(str) + '_' + X[cols[2]].astype(str)

        return X

def return_var_names(x, type=1):
    names = []
    for i in range(len(x)):
        if type == 1:
            names.append('var'+str(x[i]) + '_Categorify')
        if type == 2:
            names.append('var'+str(x[i]))
 
    return tuple(names)

list_combined_idx_cat1 = [(1,7), (1,20),(7,8),(7,20),(7,23),(7,28),(7,29), (7,39),(1,7,8),(1,7,14),(1,7,20),(1,7,23),(1,7,28),(1,7,29),(1,7,31),(1,7,39),(1,8,20),(1,8,23),(1,20,23),(1,20,28),(1,20,29),(1,20,39),(1,23,28),(1,7,14),(1,7,23),(1,7,28),(1,7,29),(1,7,31),(1,7,39),(1,8,20),(1,8,23)]
list_combined_idx_cat2 = [(1,7), (1,20),(7,8),(7,20),(7,23),(7,28),(7,29), (7,39),(1,7,8),(1,7,14),(1,7,20),(1,7,23),(1,7,28),(1,7,29),(1,7,31),(1,7,39),(1,8,20),(1,8,23),(1,20,23),(1,20,28),(1,20,29),(1,20,39),(1,23,28)]
list_combined_cat1 = [return_var_names(x, type=1) for x in list_combined_idx_cat1]
list_combined_cat2 = [return_var_names(x, type=2) for x in list_combined_idx_cat2]
combined_columns_cat1 = [f"{x[0]}_{x[1]}" if len(x) == 2 else f"{x[0]}_{x[1]}_{x[2]}"for x in list_combined_cat1]
combined_columns_cat2 = [f"{x[0]}_{x[1]}" if len(x) == 2 else f"{x[0]}_{x[1]}_{x[2]}"for x in list_combined_cat2]

def make_pipeline(parameters):
    params = Parameters(parameters)
    list_remove1 = [44, 45, 46, 47, 48, 49, 50, 51, 55, 62, 63, 64, 65, 66, 31]
    list_remove1 = ['var'+str(x) for x in list_remove1]
    list_remove2 = [10,11,12,13,15,16,17,18,22,24,25,26,27,30,31,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,53,57,59,60,61,62,63,64,65,66,67,68]
    list_remove2 = ['var'+str(x) for x in list_remove2]

    if params.approach == 1:
        
        return pipeline_1(params, list_remove1)

    if params.approach == 2:

        return pipeline_2(params, list_remove1)
    
    if params.approach == 3:

        return pipeline_3(params, list_remove1)
    
    if params.approach == 4:

        return pipeline_4(params, list_remove1)

    if params.approach == 5:

        return pipeline_5(params, list_remove1)

    if params.approach == 6:

        return pipeline_5(params, list_remove2)

    if params.approach == 7:

        return pipeline_4(params, list_remove2)

def pipeline_1(params, list_remove):
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])
    num_dis = remove_feature(list_remove, params.num_dis)
    num_con = remove_feature(list_remove, params.num_con)
    cat_nom = remove_feature(list_remove, params.cat_nom)
    cat_ord = remove_feature(list_remove, params.cat_ord)
    all_columns = num_dis + num_con + cat_nom + cat_ord

    full_pipeline = ColumnTransformer([
        ("replece_num_to_median", SimpleImputer(missing_values=-999, strategy="median"), all_columns),
        ("num", num_pipeline, num_dis + num_con),
        ("non_num", OneHotEncoder(), cat_nom + cat_ord)
    ])

    return full_pipeline, list_remove


def pipeline_2(params, list_remove):
    num_dis = remove_feature(list_remove, params.num_dis)
    num_con = remove_feature(list_remove, params.num_con)
    cat_nom = remove_feature(list_remove, params.cat_nom)
    cat_ord = remove_feature(list_remove, params.cat_ord)
    all_columns = num_dis + num_con + cat_nom + cat_ord
    categorify_col = [col + '_Categorify' for col in cat_nom + cat_ord]

    cat_pipeline = Pipeline([
        ('missing_values_cat', ProcessMissingValues1(cat_columns=categorify_col, columns=cat_nom+cat_ord)),                       
        ('categorify', Categorify1(columns=cat_nom+cat_ord, freq_treshhold=5)),
        ('get_rid_categorical_features', GetRidCategoricalFeatures(columns=cat_nom+cat_ord)), 

    ]) 

    num_pipeline = Pipeline([                     
        ('missing_values_num', ProcessMissingValues1(cat_columns=cat_nom+cat_ord,
                                                    columns=num_dis+num_con, 
                                                    type_columns='numerical', 
                                                    num_imputer=None)),
        ('standard', StandardNumerical()) 
    ]) 

    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline, all_columns),
        ("num", num_pipeline, all_columns),
        ("non_num", OneHotEncoder(), cat_nom + cat_ord)],
        remainder='passthrough')

    return full_pipeline, list_remove


def pipeline_3(params, list_remove):
    num_dis = remove_feature(list_remove, params.num_dis)
    num_con = remove_feature(list_remove, params.num_con)
    all_columns = num_dis + num_con + params.cat_nom + params.cat_ord
    list_remove = list_remove + [col + '_Categorify' for col in list_remove]
    categorify_col = [col + '_Categorify' for col in params.cat_nom + params.cat_ord]
    cat_pipeline = Pipeline([
        ('missing_values_cat', ProcessMissingValues1(cat_columns=categorify_col, columns=params.cat_nom+params.cat_ord)),                       
        ('categorify1', Categorify1(columns=params.cat_nom+params.cat_ord, freq_treshhold=5)),
        ('combining_columns_cat', CombiningColumns(list_combined=list_combined_cat1)),
        ('categorify2', Categorify1(columns=combined_columns_cat1, freq_treshhold=5)),
        ('get_rid_categorical_features', GetRidCategoricalFeatures(columns=params.cat_nom+params.cat_ord+list_remove+combined_columns_cat1))
    
    ]) 

    num_pipeline = Pipeline([                     
        ('missing_values_num', ProcessMissingValues1(cat_columns=params.cat_nom+params.cat_ord,
                                                    columns=num_dis+num_con, 
                                                    type_columns='numerical', 
                                                    num_imputer=None)),
        ('standard', StandardNumerical()) 
    ]) 
    
    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline, all_columns),
        ("num", num_pipeline, all_columns),
        ("non_num", OneHotEncoder(), params.cat_nom + params.cat_ord)],
        remainder='passthrough')
    
    return full_pipeline, list_remove

def pipeline_4(params, list_remove):
    num_dis = remove_feature(list_remove, params.num_dis)
    num_con = remove_feature(list_remove, params.num_con)
    cat_nom = remove_feature(list_remove, params.cat_nom)
    cat_ord = remove_feature(list_remove, params.cat_ord)
    categorify_col = [col + '_Categorify' for col in cat_nom + cat_ord]
    all_columns = num_dis + num_con + cat_nom + cat_ord

    cat_pipeline = Pipeline([
        ('missing_values_cat', ProcessMissingValues2(cat_columns=categorify_col, columns=cat_nom+cat_ord)),                     
        ('categorify', Categorify2(columns=cat_nom+cat_ord, freq_treshhold=5)),
        ('get_rid_categorical_features', GetRidCategoricalFeatures(columns=cat_nom+cat_ord)), 

    ]) 

    num_pipeline = Pipeline([                     
        ('missing_values_num', ProcessMissingValues2(cat_columns=cat_nom+cat_ord,
                                                    columns=num_dis+num_con, 
                                                    type_columns='numerical', 
                                                    num_imputer=None)),
        ('standard', StandardNumerical()) 
    ]) 

    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline, all_columns),
        ("num", num_pipeline, all_columns),
        ("non_num", OneHotEncoder(), cat_nom + cat_ord)],
        remainder='passthrough')

    return full_pipeline, list_remove

def pipeline_5(params, list_remove):
    list_remove = list_remove + [col + '_Categorify' for col in list_remove]
    num_dis = remove_feature(list_remove, params.num_dis)
    num_con = remove_feature(list_remove, params.num_con)
    all_columns = num_dis + num_con + params.cat_nom + params.cat_ord
    categorify_col = [col + '_Categorify' for col in remove_feature(list_remove, params.cat_nom+params.cat_ord)]
    
    cat_pipeline = Pipeline([
        ('missing_values_cat', ProcessMissingValues2(cat_columns=categorify_col, columns=params.cat_nom+params.cat_ord)),                       
        ('categorify1', Categorify2(columns=params.cat_nom+params.cat_ord, freq_treshhold=5)),
        ('combining_columns_cat', CombiningColumns(list_combined=list_combined_cat2)),
        ('categorify2', Categorify2(columns=combined_columns_cat2, freq_treshhold=5)),
        ('get_rid_categorical_features', GetRidCategoricalFeatures(columns=list_remove))
    
    ]) 
    
    num_pipeline = Pipeline([                     
        ('missing_values_num', ProcessMissingValues1(cat_columns=params.cat_nom+params.cat_ord,
                                                    columns=num_dis+num_con, 
                                                    type_columns='numerical', 
                                                    num_imputer=None)),
        ('standard', StandardNumerical()) 
    ]) 
    
    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline, all_columns),
        ("num", num_pipeline, all_columns),
        ("non_num", OneHotEncoder(), remove_feature(list_remove, num_dis) + combined_columns_cat2)],
        remainder='passthrough')

    return full_pipeline, (list_remove, combined_columns_cat2)

    

class Parameters:
    def __init__(self, params):
        self.approach = params["approach"]
        self.num_dis = params["num_dis"]
        self.num_con = params["num_con"]
        self.cat_nom = params["cat_nom"]
        self.cat_ord = params["cat_ord"]