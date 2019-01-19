# import pandas as pd
import datetime
import logging
# import argparse
import subprocess
import os
import sys
this_folder = '/user01'
cwd = os.getcwd()
sys.path.append(cwd.replace(this_folder, ''))
from models.kfold_lgbm import kfold_lightgbm
from models.kfold_xgb import kfold_xgb
from utils import load_datasets, make_output_dir, display_importances  # , save2pkl, line_notify, submit, load_target

# config
create_features = False  # create_features.py を再実行する場合は True, そうでない場合は False
is_debug = True  # True だと少数のデータで動かします, False だと全データを使います. また NUM_FOLDS = 2 になります
use_GPU = False
COMPETITION_NAME = 'elo-merchant-category-recommendation'  # submit用のapi (from utils import submit) を使うときのみ使用
FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'Outlier_Likelyhood', 'OOF_PRED', 'outliers_pred', 'month_0']
NUM_FOLDS = 11  # is_debug が True だとここの設定によらず 2 に更新されます
LOSS = 'rmse'


# start log
now = datetime.datetime.now()
logging.basicConfig(
    filename='../logs/log_{0:%Y-%m-%d-%H-%M-%S}.log'.format(now),
    level=logging.DEBUG
)
logging.debug('../logs/log_{0:%Y-%m-%d-%H-%M-%S}.log'.format(now))

# create features
if create_features:
    result = subprocess.run('python create_features.py', shell=True)
    if result.returncode != 0:
        print('ERROR: create_features.py')
        quit()

# loading
path = cwd.replace(this_folder, '/features')
train_df, test_df = load_datasets(path)

# debug or not
if is_debug:
    train_df = train_df.iloc[0:10000]
    test_df = test_df.iloc[0:10000]
    NUM_FOLDS = 2
logging.debug("Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

# model
models, model_params, feature_importance_df, train_preds, test_preds, scores = kfold_lightgbm(
    train_df, test_df, model_loss=LOSS, num_folds=NUM_FOLDS,
    feats_exclude=FEATS_EXCLUDED, stratified=False, use_gpu=use_GPU)
"""
models, model_params, feature_importance_df, train_preds, test_preds, scores = kfold_xgb(
    train_df, test_df, model_loss=LOSS, num_folds=NUM_FOLDS,
    feats_exclude=FEATS_EXCLUDED, stratified=False, use_gpu=use_GPU)
"""

# CVスコア
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)
logging.debug('===CV scores===')
logging.debug(scores)
logging.debug(score)

# submitファイルなどをまとめて保存します
folder_path = make_output_dir(score)
test_df.loc[:, 'target'] = test_preds
test_df = test_df.reset_index()
test_df[['card_id', 'target']].to_csv(
    '{0}/submit_{1:%Y-%m-%d-%H-%M-%S}_{2}.csv'.format(folder_path, now, score),
    index=False
)
train_df.loc[:, 'OOF_PRED'] = train_preds
train_df = train_df.reset_index()
train_df[['card_id', 'OOF_PRED']].to_csv(
    '{0}/oof.csv'.format(folder_path),
)
display_importances(feature_importance_df,
                    '{}/lgbm_importances.png'.format(folder_path),
                    '{}/feature_importance_lgbm.csv'.format(folder_path))

print('finish')
