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


def log_best_lgbm(model, metric):
    logging.debug('best iteration:{}'.format(model.best_iteration))
    logging.debug('best score:{}'.format(model.best_score['valid'][metric]))


def log_best_xgb(model):
    logging.debug('best iteration:{}'.format(model.best_iteration))
    logging.debug('best score:{}'.format(model.best_score))


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


def create_score_log(scores):
    score = sum(scores) / len(scores)
    print('===CV scores===')
    print(scores)
    print(score)
    logging.debug('===CV scores===')
    logging.debug(scores)
    logging.debug(score)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


@contextmanager
def timer(name):
    t0 = time.time()
    print('[{}] start'.format(name))
    yield
    print('[{}] done in {} s'.format(name,time.time()-t0))


def one_hot_encoder(df, nan_as_category=True):
    original_cols = list(df.columns)
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df,
                        columns=categorical_cols,
                        dummy_na=nan_as_category)
    new_cols = [c for c in df.columns if c not in original_cols]
    return df, new_cols


def load_datasets(feature_path, is_debug=False):
    # dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    feats = [f for f in os.listdir(feature_path) if f[-13:] == 'train.feather']
    dfs = [feather.read_dataframe(feature_path+'/'+f) for f in feats]
    train = pd.concat(dfs, axis=1)
    # dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    feats = [f for f in os.listdir(feature_path) if f[-12:] == 'test.feather']
    dfs = [feather.read_dataframe(feature_path+'/'+f) for f in feats]
    test = pd.concat(dfs, axis=1)
    if is_debug:
        train = train.iloc[0:10000,:]
        test = test.iloc[0:10000,:]
    return train, test


# 欠損値の率が高い変数を発見する機能
def findMissingColumns(data, threshold):
    missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
    col_missing = missing.index[missing > threshold]
    col_missing = [column for column in col_missing]
    return col_missing


def removeMissingColumns(train_df, test_df, threshold_col):
    missing_cols = findMissingColumns(train_df, threshold_col)
    train_df = train_df.drop(missing_cols, axis=1)
    test_df = test_df.drop(missing_cols, axis=1)
    logging.debug('missing cols:{}'.format(missing_cols))
    return train_df, test_df


def line_notify(message):
    f = open('../data/input/line_token.txt')
    token = f.read()
    f.close
    line_notify_token = token.replace('\n', '')
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    requests.post(line_notify_api, data=payload, headers=headers)
    print(message)


# https://github.com/KazukiOnodera/Home-Credit-Default-Risk/blob/master/py/utils.py
def submit(competition_name, file_path, comment='from API'):
    os.system(
        'kaggle competitions submit -c {} -f {} -m "{}"'
        .format(competition_name, file_path, comment))
    time.sleep(60)  # tekito~~~~
    tmp = os.popen(
        'kaggle competitions submissions -c {} -v | head -n 2'
        .format(competition_name)).read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i, j in zip(col.split(','), values.split(',')):
        message += '{}: {}\n'.format(i, j)
#        print(f'{i}: {j}') # TODO: comment out later?
    line_notify(message.rstrip())


def save2pkl(path, object):
    f = open(path, 'wb')
    pickle.dump(object, f)
    f.close


def loadpkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    f.close
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
def save_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]]\
        .groupby("feature").mean()\
        .sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_\
        .loc[feature_importance_df_.feature.isin(cols)]

    # importance下位の確認用に追加しました
    _feature_importance_df_ = feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)


def make_output_dir(score, model_name):
    path = '../data/output/'
    folders = []
    for x in os.listdir(path):
        if os.path.isdir(path + x):
            folders.append(x)
    if folders == []:
        folder_name = '/001_{0}_score_{1}'.format(model_name, score)
    else:
        max_folder_num = max(int(f[:3]) for f in folders)
        folder_name = '/{0:0=3}_{1}_score_{2}'.format(max_folder_num+1, model_name, score)
    os.mkdir(path + folder_name)
    return path + folder_name


# この部分は上記の各関数の動作確認のために使います
if __name__ == '__main__':
    make_output_dir(0.02, 'lgbm')
