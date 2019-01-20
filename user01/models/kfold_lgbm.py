import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import logging
import pandas as pd
import numpy as np
import os
import sys
this_folder = '/user01/models'
cwd = os.getcwd()
sys.path.append(cwd.replace(this_folder, ''))
from utils import log_best_lgbm


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(train_df, test_df, model_loss, num_folds, feats_exclude, stratified=False, use_gpu=False):
    model_name = sys._getframe().f_code.co_name
    logging.debug("Starting {}".format(model_name))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=4950)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=4950)

    # Create arrays and dataframes to store results
    train_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in feats_exclude]

    models = []
    model_params = {}
    scores = []

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        print('Fold_{}'.format(n_fold))
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
        lgb_valid = lgb.Dataset(valid_x, label=valid_y, free_raw_data=False)

        # パラメータは適当です
        params = {
                'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': model_loss,
                'learning_rate': 0.01,
                'subsample': 0.9855232997390695,
                'max_depth': 7,
                'top_rate': 0.9064148448434349,
                'num_leaves': 63,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed': int(2**n_fold),
                'bagging_seed': int(2**n_fold),
                'drop_seed': int(2**n_fold)
                }
        if use_gpu:
            params.update({'device': 'gpu'})

        model = lgb.train(
                         params,
                         lgb_train,
                         valid_sets=[lgb_train, lgb_valid],
                         valid_names=['train', 'valid'],
                         num_boost_round=10000,
                         early_stopping_rounds=200,
                         verbose_eval=100
                         )

        # save model
        models.append(model)
        model_params.update({str(n_fold): params})

        train_preds[valid_idx] = model.predict(valid_x, num_iteration=model.best_iteration)
        test_preds += model.predict(test_df[feats], num_iteration=model.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(
            model.feature_importance(importance_type='gain', iteration=model.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)
        logging.debug('Fold_{}_best'.format(n_fold))
        log_best_lgbm(model, model_loss)
        scores.append(model.best_score['valid'][model_loss])

    return models, model_params, feature_importance_df, train_preds, test_preds, scores, model_name
