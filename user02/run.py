# import pandas as pd
import datetime
import logging
# import argparse
import subprocess
import json
import numpy as np
import os
import pandas as pd
import sys
this_folder = '/user02'
cwd = os.getcwd()
sys.path.append(cwd.replace(this_folder, ''))
from models.kfold_lgbm import kfold_lightgbm
from models.kfold_xgb import kfold_xgb
from utils import load_datasets, create_score_log, make_output_dir, save_importances, save2pkl # , line_notify, submit, load_target
from utils import submit, line_notify

# config
create_features = True  # create_features.py を再実行する場合は True, そうでない場合は False
is_debug = False  # True だと少数のデータで動かします, False だと全データを使います. また folds = 2 になります
use_GPU = True

competition_name = 'elo-merchant-category-recommendation'
target_col = 'target'
feats_exclude = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'Outlier_Likelyhood', 'OOF_PRED', 'outliers_pred', 'month_0']
folds = 11 if not is_debug else 2  # is_debug が True なら2, そうでなければ11
loss_type = 'rmse'


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
feats = json.load(open('features_to_use.json'))['features']
train_df, test_df = load_datasets(path, is_debug)
train_df, test_df = train_df[feats], test_df[feats]
logging.debug("Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

# model
models, model_params, feature_importance_df, train_preds, test_preds, scores, model_name = kfold_lightgbm(
    train_df, test_df, target_col=target_col, model_loss=loss_type,
    num_folds=folds, feats_exclude=feats_exclude, stratified=False, use_gpu=use_GPU)
"""
models, model_params, feature_importance_df, train_preds, test_preds, scores, model_name = kfold_xgb(
    train_df, test_df, target_col=target_col, model_loss=loss_type,
    num_folds=folds, feats_exclude=feats_exclude, stratified=False, use_gpu=use_GPU)
"""
# CVスコア
create_score_log(scores)
score = np.mean(np.array(scores))
line_notify('Full RMSE score %.6f' % score)

# submitファイルなどをまとめて保存します. ほんとはもっと疎結合にしてutilに置けるようにしたい...
def output(train_df, test_df, models, model_params, feature_importance_df, train_preds, test_preds, scores, now, model_name):
    score = sum(scores) / len(scores)
    folder_path = make_output_dir(score, now, model_name)
    for i, m in enumerate(models):
        save2pkl('{0}/model_{1:0=2}.pkl'.format(folder_path, i), m)
    with open('{0}/model_params.json'.format(folder_path), 'w') as f:
        json.dump(model_params, f, indent=4)
    with open('{0}/model_valid_scores.json'.format(folder_path), 'w') as f:
        json.dump({i:s for i, s in enumerate(scores)}, f, indent=4)
    save_importances(
        feature_importance_df,
        '{}/importances.png'.format(folder_path),
        '{}/importance.csv'.format(folder_path))

    # 以下の部分はコンペごとに修正が必要
    submission_file_name = '{0}/submit_{1:%Y-%m-%d-%H-%M-%S}_{2}.csv'.format(folder_path, now, score)

    test_df.loc[:, 'target'] = test_preds
    test_df = test_df.reset_index()
    test_df[['card_id', 'target']].to_csv(submission_file_name,index=False)

    train_df.loc[:, 'OOF_PRED'] = train_preds
    train_df = train_df.reset_index()
    train_df[['card_id', 'OOF_PRED']].to_csv('{0}/oof.csv'.format(folder_path),)

    # API経由でsubmit
    if not is_debug:
        submit(competition_name, submission_file_name, comment='model101 cv: %.6f' % score)

output(train_df, test_df, models, model_params, feature_importance_df, train_preds, test_preds, scores, now, model_name)
