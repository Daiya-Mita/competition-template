import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import numpy as np
import os
import gc
import sys
this_folder = '/user01/models'
cwd = os.getcwd()
sys.path.append(cwd.replace(this_folder, ''))
from utils import log_evaluation, log_best_xgb


# LightGBM GBDT with KFold or Stratified KFold
def kfold_xgb(train_df, test_df, model_loss, num_folds, feats_exclude, stratified = False, use_gpu = False):
    logging.debug("Starting XGBoost.")

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=4950)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=4950)

    # Create arrays and dataframes to store results
    train_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in feats_exclude]

    # final predict用にdmatrix形式のtest dfを作っておきます
    test_df_dmtrx = xgb.DMatrix(test_df[feats])

    models = []
    model_params = {}
    scores = []

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        print('Fold_{}'.format(n_fold))
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        xgb_train = xgb.DMatrix(train_x,
                                label=train_y)
        xgb_valid = xgb.DMatrix(valid_x,
                               label=valid_y)

        # params
        params = {
                'booster': 'gbtree',
                'eval_metric':model_loss,
                'silent':1,
                'eta': 0.01,
                'max_leaves': 63,
                'colsample_bytree': 0.5665320670155495,
                'subsample': 0.9855232997390695,
                'max_depth': 7,
                'reg_alpha': 9.677537745007898,
                'reg_lambda': 8.2532317400459,
                'gamma': 9.820197773625843,
                'min_child_weight': 41.9612869171337,
                'seed':int(2**n_fold)
                }
        if use_gpu == True:
            params.update({'objective':'gpu:reg:linear',
                           'tree_method': 'gpu_hist',
                           'predictor': 'gpu_predictor'})

        model = xgb.train(
                         params,
                         xgb_train,
                         num_boost_round=10000,
                         evals=[(xgb_train,'train'),(xgb_valid,'valid')],
                         early_stopping_rounds= 200,
                         verbose_eval=100
                         )

        # save model
        models.append(model)
        model_params.update({str(n_fold):params})

        train_preds[valid_idx] = model.predict(xgb_valid)
        test_preds += model.predict(test_df_dmtrx) / folds.n_splits

        fold_importance_df = pd.DataFrame.from_dict(model.get_score(importance_type='gain'), orient='index', columns=['importance'])
        fold_importance_df["feature"] = fold_importance_df.index.tolist()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        logging.debug('Fold_{}_best'.format(n_fold))
        log_best_xgb(model, model_loss)
        scores.append(model.best_score)

    return models, model_params, feature_importance_df, train_preds, test_preds, scores
