import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Important parameters
# --------------------------------------------------------------------------------------------------------
def return_var_names(x):
  names = []
  for i in range(len(x)):
    names.append('var'+str(x[i]) + '_Categorify')

  return tuple(names)

def unique_index(l_comb):
  index = []
  for x in l_comb:
    index += list(x)
  return list(set(index))

categorical_feature_chosen = "var13"
PATH = "/content/drive/MyDrive/Colab Notebooks/Porto_Seguro_competition/"
list_remove = [44, 45, 46, 47, 48, 49, 50, 51, 55, 62, 63, 64, 65, 66] #31]
list_remove = ['var'+str(x) for x in list_remove]
list_combined_idx = [(1,7), (1,20),(7,8),(7,20),(7,23),(7,28),(7,29), (7,39),
        (1,7,8),(1,7,14),(1,7,20),(1,7,23),(1,7,28),(1,7,29),(1,7,31),(1,7,39),
        (1,8,20),(1,8,23),(1,20,23),(1,20,28),(1,20,29),(1,20,39),(1,23,28),
        (2,3,8),(1,7,14),(1,7,23),(1,7,28),(1,7,29),(1,7,31),(1,7,39),(1,8,20),
        (1,8,23)]
list_combined = [return_var_names(x) for x in list_combined_idx]
columns_name = []


# Classes
# --------------------------------------------------------------------------------------------------------

class ProcessMissingValues(BaseEstimator, TransformerMixin):
  def __init__(self, 
               columns,
               categorical_features_ignore,
               type_columns='categorical',
               categorical_feature_chosen="var31",
               replace_num_to_median=False):
    self.columns = columns
    self.categorical_features_ignore = categorical_features_ignore
    self.type_columns = type_columns
    self.categorical_feature_chosen = categorical_feature_chosen
    self.replace_num_to_median=replace_num_to_median

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
    self.columns = [col for col in X.columns if 'NA_' not in col and col not in self.categorical_features_ignore]

    if self.replace_num_to_median:
      for col in self.columns:
        X[col] = X[col].replace(-999, np.nan)
        median = X[col].median()
        if 'TE_' not in col and 'CE_' not in col:
          X['NA_' + col] = X[col].isna().astype(np.int8)
        X = X.fillna(median)

    else:
      for col in self.columns:
        X[col] = X[col].replace(-999, np.nan)
        X_median = X[[self.categorical_feature_chosen, col]] \
                    .groupby(self.categorical_feature_chosen).median() \
                    .reset_index()
        col_name = col + '_median_per_' + self.categorical_feature_chosen
        X_median.columns = [self.categorical_feature_chosen, col_name]
        X = X.merge(X_median, how='left', on=self.categorical_feature_chosen)
        if 'TE_' not in col and 'CE_' not in col:
          X['NA_' + col] = X[col].isna().astype(np.int8)
        X.loc[X[col].isna(), col] = X.loc[X[col].isna(), col_name]
        X.drop(col_name, axis=1, inplace=True)

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
      X = X.merge(self.freqs[idx], how='left', on=col)

    return X

  def make_columns(self, X):
    self.freqs = []
    for col in self.columns:
      freq = X[col].value_counts()
      freq = freq.reset_index()
      freq.columns = [col, 'count']
      freq = freq.reset_index()
      freq.columns = [col + '_Categorify', col, 'count']
      freq[col + '_Categorify'] = freq[col + '_Categorify']+2
      freq.loc[freq['count']<self.freq_treshhold, col + '_Categorify'] = self.lowfrequency_id
      freq.loc[freq[col]=='UNKNOWN', col + '_Categorify'] = self.unkown_id
      freq = freq.drop('count', axis=1)
      self.freqs.append(freq)

class GetRidCategoricalFeatures(BaseEstimator, TransformerMixin):
  def __init__(self, columns, isOneHot=False, combined_columns=[]):
    self.columns_ = columns
    self.columns = columns + combined_columns
    self.isOneHot = isOneHot

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    if self.isOneHot:
      cat_categorify = [col + "_Categorify" for col in self.columns_]
      cat_cols = [x for x in X.columns if '_Categorify_Categorify' in x or x in cat_categorify]
      self.columns += cat_cols
    for col in self.columns:
      if self.isOneHot:
        if col in X.columns:
          X.drop(col, axis=1,inplace=True)
      else:
        categority_col = col + '_Categorify'
        X[[col]] = X[[categority_col]].copy()
        X.drop(categority_col, axis=1,inplace=True)

    return X

class StandardNumerical(BaseEstimator, TransformerMixin):
  def __init__(self, columns, isOneHot=False):
    self.columns_ = columns
    self.columns = []
    self.isOneHot = isOneHot

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    if self.isOneHot:
      self.columns = self.columns_
    else:
      self.columns = [col for col in X.columns if "NA" not in col]
    for col in self.columns:
      X[col] = X[col].replace(-999, np.nan)
      median = X[col].median()
      X.loc[X[col].isna(), col] = median
      X[col]= (X[col] - np.mean(X[col]))/np.std(X[col])

    return X

class OneHotEncoding(BaseEstimator, TransformerMixin):
  def __init__(self, columns, add_features=[], remove_features=[]):
    self.columns = [col + "_Categorify" for col in columns] + add_features
    self.remove_features = remove_features
    self.x_cat = None

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X.drop(self.remove_features, axis=1, inplace=True)
    self.columns = [x for x in self.columns if x not in self.remove_features]
    out = []
    for col in self.columns:
      out.append(pd.get_dummies(X[col], prefix=col))
    out.append(X)
    X = pd.concat(out, axis=1)
    return X

  def make_columns(self,categories):
    name_columns = []
    for idx, col in enumerate(categories):
      for var in col:
        name = "{}_{}".format(self.columns[idx], var)
        name_columns.append(name)
    return name_columns

class CombiningColumns(BaseEstimator, TransformerMixin):
  def __init__(self, list_combined=list_combined):
    self.list_combined = list_combined

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    name_col = []
    for cols in self.list_combined:
      if len(cols) == 2:
        name_col.append(cols[0] + '_' + cols[1])
        X[name_col[-1]] = X[cols[0]].astype(str) + "_" +  X[cols[1]].astype(str)
      else:
        name_col.append(cols[0] + '_' + cols[1] + '_' + cols[2])
        X[name_col[-1]] = X[cols[0]].astype(str) + "_" +  X[cols[1]].astype(str) + '_' + X[cols[2]].astype(str)

    return X

combined_columns = [f"{x[0]}_{x[1]}" if len(x) == 2 else f"{x[0]}_{x[1]}_{x[2]}"for x in list_combined]

# Pipeline
# --------------------------------------------------------------------------------------------------------

def make_pipeline(cat_nom, cat_ord, num_con, num_dis):
    pipeline = Pipeline([
        ('missing_values_categorical', ProcessMissingValues(columns=cat_nom, 
                                                            categorical_features_ignore=cat_nom+cat_ord)),
        ('missing_values_numerical', ProcessMissingValues(columns=cat_nom,
                                                            categorical_features_ignore=cat_nom+cat_ord,
                                                            type_columns='numerical',
                                                            categorical_feature_chosen=categorical_feature_chosen,
                                                            replace_num_to_median=True)),
        ('categorify', Categorify(columns=cat_nom)),
        ('combiningColumns', CombiningColumns()),
        ('categorify_combined', Categorify(columns=combined_columns)),
        ('one_hot_encoding', OneHotEncoding(add_features=combined_columns, columns=cat_nom, remove_features=unique_index(list_combined))),
        ('get_rid_categorical_features', GetRidCategoricalFeatures(isOneHot=True, 
                                                                    columns=cat_nom,
                                                                    combined_columns=combined_columns)),
        ('standard', StandardNumerical(isOneHot=True, columns=num_con + num_dis + cat_ord))               
    ])

    return pipeline