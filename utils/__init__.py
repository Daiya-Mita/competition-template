import time
import pandas as pd
import numpy as np
import feather
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os
from contextlib import contextmanager
import pickle
import logging
from lightgbm.callback import _format_eval_result
from sklearn.metrics import mean_squared_error


def log_best(model, metric):
    logging.debug(model.best_iteration)
    logging.debug(model.best_score['valid'][metric])


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list \
                and (env.iteration + 1) % period == 0:
            result = '\t'.join([
                _format_eval_result(x, show_stdv)
                for x in env.evaluation_result_list
            ])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback


# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def one_hot_encoder(df, nan_as_category=True):
    original_cols = list(df.columns)
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df,
                        columns=categorical_cols,
                        dummy_na=nan_as_category)
    new_cols = [c for c in df.columns if c not in original_cols]
    return df, new_cols


def load_datasets(features_path):
    # dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    train_features = [f for f in os.listdir(features_path) if f[-13:] == 'train.feather']
    dfs = [feather.read_dataframe(features_path+'/'+f) for f in train_features]
    train = pd.concat(dfs, axis=1)
    # dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    test_features = [f for f in os.listdir(features_path) if f[-12:] == 'test.feather']
    dfs = [feather.read_dataframe(features_path+'/'+f) for f in test_features]
    test = pd.concat(dfs, axis=1)
    return train, test

"""
def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    return y_train
"""

def line_notify(message):
    f = open('../data/input/line_token.txt')
    token = f.read()
    f.close
    line_notify_token = token.replace('\n', '')
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
    print(message)

# API経由でsubmitする機能 https://github.com/KazukiOnodera/Home-Credit-Default-Risk/blob/master/py/utils.py
def submit(competition_name, file_path, comment='from API'):
    os.system('kaggle competitions submit -c {} -f {} -m "{}"'.format(competition_name,file_path,comment))
    time.sleep(60) # tekito~~~~
    tmp = os.popen('kaggle competitions submissions -c {} -v | head -n 2'.format(competition_name)).read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i,j in zip(col.split(','), values.split(',')):
        message += '{}: {}\n'.format(i,j)
#        print(f'{i}: {j}') # TODO: comment out later?
    line_notify(message.rstrip())


def save2pkl(path, object):
    f = open(path, 'wb')
    pickle.dump(object, f)
    f.close


def loadpkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    return out


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    int8_min = np.iinfo(np.int8).min
    int8_max = np.iinfo(np.int8).max
    int16_min = np.iinfo(np.int16).min
    int16_max = np.iinfo(np.int16).max
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max
    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max
    float16_min = np.finfo(np.float16).min
    float16_max = np.finfo(np.float16).max
    float32_min = np.finfo(np.float32).min
    float32_max = np.finfo(np.float32).max
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > int8_min and c_max < int8_max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > int16_min and c_max < int16_max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > int32_min and c_max < int32_max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > int64_min and c_max < int64_max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > float16_min and c_max < float16_max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > float32_min and c_max < float32_max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    mem_diff_pct = (start_mem - end_mem) / start_mem
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * mem_diff_pct))
    return df

# Display/plot feature importance
def display_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    # importance下位の確認用に追加しました
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)