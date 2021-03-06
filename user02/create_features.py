import pandas as pd
import numpy as np
import datetime
import feather
import gc
import json
from workalendar.america import Brazil
import os
import sys
this_folder = '/user02'
cwd = os.getcwd()
sys.path.append(cwd.replace(this_folder, ''))
from utils import one_hot_encoder, load_datasets
sys.path.append(cwd.replace(this_folder, '/src'))
from feature_base import Feature, generate_features

# featureの格納場所はfeatures
Feature.dir = '../features'


class Traintest(Feature):
    def create_features(self):
        # load csv
        # train_df = pd.read_csv('../input/train.csv', index_col=['card_id'], nrows=num_rows)
        train_df = feather.read_dataframe('../data/input/train.feather')
        train_df = train_df.set_index('card_id')
        # test_df = pd.read_csv('../input/test.csv', index_col=['card_id'], nrows=num_rows)
        test_df = feather.read_dataframe('../data/input/test.feather')
        test_df = test_df.set_index('card_id')

        print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

        # outlier
        train_df['outliers'] = 0
        train_df.loc[train_df['target'] < -30, 'outliers'] = 1

        # set target as nan
        test_df['target'] = np.nan

        # merge
        df = train_df.append(test_df)

        del train_df, test_df
        gc.collect()

        # datetimeへ変換
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])

        # datetime features
    #    df['month'] = df['first_active_month'].dt.month.fillna(0).astype(int).astype(object)
    #    df['year'] = df['first_active_month'].dt.year.fillna(0).astype(int).astype(object)
    #    df['dayofweek'] = df['first_active_month'].dt.dayofweek.fillna(0).astype(int).astype(object)
    #    df['weekofyear'] = df['first_active_month'].dt.weekofyear.fillna(0).astype(int).astype(object)
        df['quarter'] = df['first_active_month'].dt.quarter
    #    df['month_year'] = df['month'].astype(str)+'_'+df['year'].astype(str)
        df['elapsed_time'] = (pd.to_datetime('2019-01-18') - df['first_active_month']).dt.days

        df['days_feature1'] = df['elapsed_time'] * df['feature_1']
        df['days_feature2'] = df['elapsed_time'] * df['feature_2']
        df['days_feature3'] = df['elapsed_time'] * df['feature_3']

        df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
        df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
        df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

        # one hot encoding
        df, cols = one_hot_encoder(df, nan_as_category=False)

        for f in ['feature_1','feature_2','feature_3']:
            order_label = df.groupby([f])['outliers'].mean()
            df[f] = df[f].map(order_label)

        df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
        df['feature_mean'] = df['feature_sum']/3
        df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
        df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
        df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

        # df = df.reset_index()
        # self.df = df
        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]
        self.train = train_df.reset_index()
        self.test = test_df.reset_index()


class Historical_transactions(Feature):
    def create_features(self):
        #hist_df = pd.read_csv('../input/historical_transactions.csv', nrows=num_rows)
        hist_df = feather.read_dataframe('../data/input/historical_transactions.feather')

        # fillna
        hist_df['category_2'].fillna(1.0,inplace=True)
        hist_df['category_3'].fillna('A',inplace=True)
        hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
        hist_df['installments'].replace(-1, np.nan,inplace=True)
        hist_df['installments'].replace(999, np.nan,inplace=True)

        # trim
        hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))

        # Y/Nのカラムを1-0へ変換
        hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
        hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
        hist_df['category_3'] = hist_df['category_3'].map({'A':0, 'B':1, 'C':2})

        # datetime features
        hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    #    hist_df['year'] = hist_df['purchase_date'].dt.year
        hist_df['month'] = hist_df['purchase_date'].dt.month
        hist_df['day'] = hist_df['purchase_date'].dt.day
        hist_df['hour'] = hist_df['purchase_date'].dt.hour
        hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
        hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
        hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >=5).astype(int)

        # additional features
        hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

        #ブラジルの休日
        cal = Brazil()
        hist_df['is_holiday'] = hist_df['purchase_date'].dt.date.apply(cal.is_holiday).astype(int)

        # 購入日からイベント日までの経過日数
        #Christmas : December 25 2017
        hist_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Mothers Day: May 14 2017
        hist_df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #fathers day: August 13 2017
        hist_df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Childrens day: October 12 2017
        hist_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Valentine's Day : 12th June, 2017
        hist_df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Black Friday : 24th November 2017
        hist_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        #2018
        #Mothers Day: May 13 2018
        hist_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        hist_df['month_diff'] = ((pd.to_datetime('2019-01-18') - hist_df['purchase_date']).dt.days)//30
        hist_df['month_diff'] += hist_df['month_lag']

        # additional features
        hist_df['duration'] = hist_df['purchase_amount']*hist_df['month_diff']
        hist_df['amount_month_ratio'] = hist_df['purchase_amount']/hist_df['month_diff']

        # memory usage削減
#        hist_df = reduce_mem_usage(hist_df)

        col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']
        col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

        aggs = {}
        for col in col_unique:
            aggs[col] = ['nunique']

        for col in col_seas:
            aggs[col] = ['nunique', 'mean', 'min', 'max']

        aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
        aggs['installments'] = ['sum','max','mean','var','skew']
        aggs['purchase_date'] = ['max','min']
        aggs['month_lag'] = ['max','min','mean','var','skew']
        aggs['month_diff'] = ['max','min','mean','var','skew']
        aggs['authorized_flag'] = ['mean']
        aggs['weekend'] = ['mean'] # overwrite
        aggs['weekday'] = ['mean'] # overwrit
        aggs['day'] = ['nunique', 'mean', 'min'] # overwrite
        aggs['category_1'] = ['mean']
        aggs['category_2'] = ['mean']
        aggs['category_3'] = ['mean']
        aggs['card_id'] = ['size','count']
        aggs['is_holiday'] = ['mean']
        aggs['price'] = ['sum','mean','max','min','var']
        aggs['Christmas_Day_2017'] = ['mean']
        aggs['Mothers_Day_2017'] = ['mean']
        aggs['fathers_day_2017'] = ['mean']
        aggs['Children_day_2017'] = ['mean']
        aggs['Valentine_Day_2017'] = ['mean']
        aggs['Black_Friday_2017'] = ['mean']
        aggs['Mothers_Day_2018'] = ['mean']
        aggs['duration']=['mean','min','max','var','skew']
        aggs['amount_month_ratio']=['mean','min','max','var','skew']

        for col in ['category_2','category_3']:
            hist_df[col+'_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
            hist_df[col+'_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
            hist_df[col+'_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
            hist_df[col+'_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
            aggs[col+'_mean'] = ['mean']

        hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

        # カラム名の変更
        hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
        hist_df.columns = ['hist_'+ c for c in hist_df.columns]

        hist_df['hist_purchase_date_diff'] = (hist_df['hist_purchase_date_max']-hist_df['hist_purchase_date_min']).dt.days
        hist_df['hist_purchase_date_average'] = hist_df['hist_purchase_date_diff']/hist_df['hist_card_id_size']
        hist_df['hist_purchase_date_uptonow'] = (pd.to_datetime('2019-01-18')-hist_df['hist_purchase_date_max']).dt.days
        hist_df['hist_purchase_date_uptomin'] = (pd.to_datetime('2019-01-18')-hist_df['hist_purchase_date_min']).dt.days

        #hist_df = hist_df.reset_index()

        train_df = feather.read_dataframe('../features/traintest_train.feather')
        test_df = feather.read_dataframe('../features/traintest_test.feather')
        df = pd.concat([train_df, test_df], axis=0)
        init_cols = df.columns
        hist_df = pd.merge(df, hist_df, on='card_id', how='outer')

        hist_df_train = hist_df[hist_df['target'].notnull()]
        hist_df_test = hist_df[hist_df['target'].isnull()]

        hist_df_train = hist_df_train.drop(init_cols, axis=1)
        hist_df_test = hist_df_test.drop(init_cols, axis=1)

        #self.df = hist_df
        self.train = hist_df_train.reset_index(drop=True)
        self.test = hist_df_test.reset_index(drop=True)


class New_merchant_transactions(Feature):
    def create_features(self):
        # load csv
        #new_merchant_df = pd.read_csv('../input/new_merchant_transactions.csv', nrows=num_rows)
        new_merchant_df = feather.read_dataframe('../data/input/new_merchant_transactions.feather')

        # fillna
        new_merchant_df['category_2'].fillna(1.0,inplace=True)
        new_merchant_df['category_3'].fillna('A',inplace=True)
        new_merchant_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
        new_merchant_df['installments'].replace(-1, np.nan,inplace=True)
        new_merchant_df['installments'].replace(999, np.nan,inplace=True)

        # trim
        new_merchant_df['purchase_amount'] = new_merchant_df['purchase_amount'].apply(lambda x: min(x, 0.8))

        # Y/Nのカラムを1-0へ変換
        new_merchant_df['authorized_flag'] = new_merchant_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
        new_merchant_df['category_1'] = new_merchant_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
        new_merchant_df['category_3'] = new_merchant_df['category_3'].map({'A':0, 'B':1, 'C':2}).astype(int)

        # datetime features
        new_merchant_df['purchase_date'] = pd.to_datetime(new_merchant_df['purchase_date'])
    #    new_merchant_df['year'] = new_merchant_df['purchase_date'].dt.year
        new_merchant_df['month'] = new_merchant_df['purchase_date'].dt.month
        new_merchant_df['day'] = new_merchant_df['purchase_date'].dt.day
        new_merchant_df['hour'] = new_merchant_df['purchase_date'].dt.hour
        new_merchant_df['weekofyear'] = new_merchant_df['purchase_date'].dt.weekofyear
        new_merchant_df['weekday'] = new_merchant_df['purchase_date'].dt.weekday
        new_merchant_df['weekend'] = (new_merchant_df['purchase_date'].dt.weekday >=5).astype(int)

        # additional features
        new_merchant_df['price'] = new_merchant_df['purchase_amount'] / new_merchant_df['installments']

        #ブラジルの休日
        cal = Brazil()
    #    new_merchant_df['is_holiday'] = new_merchant_df['purchase_date'].dt.date.apply(cal.is_holiday).astype(int)

        # 購入日からイベント日までの経過日数
        #Christmas : December 25 2017
        new_merchant_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Mothers Day: May 14 2017
    #    new_merchant_df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #fathers day: August 13 2017
    #    new_merchant_df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Childrens day: October 12 2017
        new_merchant_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Valentine's Day : 12th June, 2017
    #    new_merchant_df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Black Friday : 24th November 2017
        new_merchant_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        #2018
        #Mothers Day: May 13 2018
        new_merchant_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        new_merchant_df['month_diff'] = ((pd.to_datetime('2019-01-18') - new_merchant_df['purchase_date']).dt.days)//30
        new_merchant_df['month_diff'] += new_merchant_df['month_lag']

        # additional features
        new_merchant_df['duration'] = new_merchant_df['purchase_amount']*new_merchant_df['month_diff']
        new_merchant_df['amount_month_ratio'] = new_merchant_df['purchase_amount']/new_merchant_df['month_diff']

        # memory usage削減
#        new_merchant_df = reduce_mem_usage(new_merchant_df)

        col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']
        col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

        aggs = {}
        for col in col_unique:
            aggs[col] = ['nunique']

        for col in col_seas:
            aggs[col] = ['nunique', 'mean', 'min', 'max']

        aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
        aggs['installments'] = ['sum','max','mean','var','skew']
        aggs['purchase_date'] = ['max','min']
        aggs['month_lag'] = ['max','min','mean','var','skew']
        aggs['month_diff'] = ['mean','var','skew']
    #    aggs['authorized_flag'] = ['mean']
        aggs['weekend'] = ['mean']
        aggs['month'] = ['mean', 'min', 'max']
        aggs['weekday'] = ['mean', 'min', 'max']
        aggs['category_1'] = ['mean']
        aggs['category_2'] = ['mean']
        aggs['category_3'] = ['mean']
        aggs['card_id'] = ['size','count']
    #    aggs['is_holiday'] = [ 'mean']
        aggs['price'] = ['mean','max','min','var']
        aggs['Christmas_Day_2017'] = ['mean']
    #    aggs['Mothers_Day_2017'] = ['mean']
    #    aggs['fathers_day_2017'] = ['mean']
        aggs['Children_day_2017'] = ['mean']
    #    aggs['Valentine_Day_2017'] = ['mean']
        aggs['Black_Friday_2017'] = ['mean']
        aggs['Mothers_Day_2018'] = ['mean']
        aggs['duration']=['mean','min','max','var','skew']
        aggs['amount_month_ratio']=['mean','min','max','var','skew']

        for col in ['category_2','category_3']:
            new_merchant_df[col+'_mean'] = new_merchant_df.groupby([col])['purchase_amount'].transform('mean')
            new_merchant_df[col+'_min'] = new_merchant_df.groupby([col])['purchase_amount'].transform('min')
            new_merchant_df[col+'_max'] = new_merchant_df.groupby([col])['purchase_amount'].transform('max')
            new_merchant_df[col+'_sum'] = new_merchant_df.groupby([col])['purchase_amount'].transform('sum')
            aggs[col+'_mean'] = ['mean']

        new_merchant_df = new_merchant_df.reset_index().groupby('card_id').agg(aggs)

        # カラム名の変更
        new_merchant_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_merchant_df.columns.tolist()])
        new_merchant_df.columns = ['new_'+ c for c in new_merchant_df.columns]

        new_merchant_df['new_purchase_date_diff'] = (new_merchant_df['new_purchase_date_max']-new_merchant_df['new_purchase_date_min']).dt.days
        new_merchant_df['new_purchase_date_average'] = new_merchant_df['new_purchase_date_diff']/new_merchant_df['new_card_id_size']
        new_merchant_df['new_purchase_date_uptonow'] = (pd.to_datetime('2019-01-18')-new_merchant_df['new_purchase_date_max']).dt.days
        new_merchant_df['new_purchase_date_uptomin'] = (pd.to_datetime('2019-01-18')-new_merchant_df['new_purchase_date_min']).dt.days

        # memory usage削減
        #new_merchant_df = reduce_mem_usage(new_merchant_df)

        #new_merchant_df = new_merchant_df.reset_index()

        train_df = feather.read_dataframe('../features/traintest_train.feather')
        test_df = feather.read_dataframe('../features/traintest_test.feather')
        df = pd.concat([train_df, test_df], axis=0)
        init_cols = df.columns
        new_merchant_df = pd.merge(df, new_merchant_df, on='card_id', how='outer')

        new_merchant_df_train = new_merchant_df[new_merchant_df['target'].notnull()]
        new_merchant_df_test = new_merchant_df[new_merchant_df['target'].isnull()]

        new_merchant_df_train = new_merchant_df_train.drop(init_cols, axis=1)
        new_merchant_df_test = new_merchant_df_test.drop(init_cols, axis=1)

        self.train = new_merchant_df_train.reset_index(drop=True)
        self.test = new_merchant_df_test.reset_index(drop=True)

class Additional_features(Feature):
    def create_features(self):
        #df = load_all()
        path = cwd.replace(this_folder,'/features')
        df_1, df_2 = load_datasets(path)
        df = pd.concat([df_1, df_2], axis=0)
        init_cols = df.columns
        df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
        df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
        df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days
        df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

        date_features=['hist_purchase_date_max','hist_purchase_date_min',
                       'new_purchase_date_max', 'new_purchase_date_min']

        for f in date_features:
            df[f] = df[f].astype(np.int64) * 1e-9

        df['card_id_total'] = df['new_card_id_size']+df['hist_card_id_size']
        df['card_id_cnt_total'] = df['new_card_id_count']+df['hist_card_id_count']
        df['card_id_cnt_ratio'] = df['new_card_id_count']/df['hist_card_id_count']
        df['purchase_amount_total'] = df['new_purchase_amount_sum']+df['hist_purchase_amount_sum']
        df['purchase_amount_mean'] = df['new_purchase_amount_mean']+df['hist_purchase_amount_mean']
        df['purchase_amount_max'] = df['new_purchase_amount_max']+df['hist_purchase_amount_max']
        df['purchase_amount_min'] = df['new_purchase_amount_min']+df['hist_purchase_amount_min']
        df['purchase_amount_ratio'] = df['new_purchase_amount_sum']/df['hist_purchase_amount_sum']
        df['month_diff_mean'] = df['new_month_diff_mean']+df['hist_month_diff_mean']
        df['month_diff_ratio'] = df['new_month_diff_mean']/df['hist_month_diff_mean']
    #    df['month_diff_max'] = df['new_month_diff_max']+df['hist_month_diff_max']
    #    df['month_diff_min'] = df['new_month_diff_min']+df['hist_month_diff_min']
        df['month_lag_mean'] = df['new_month_lag_mean']+df['hist_month_lag_mean']
        df['month_lag_max'] = df['new_month_lag_max']+df['hist_month_lag_max']
        df['month_lag_min'] = df['new_month_lag_min']+df['hist_month_lag_min']
        df['category_1_mean'] = df['new_category_1_mean']+df['hist_category_1_mean']
    #    df['category_1_min'] = df['new_category_1_min']+df['hist_category_1_min']
        df['installments_total'] = df['new_installments_sum']+df['hist_installments_sum']
        df['installments_mean'] = df['new_installments_mean']+df['hist_installments_mean']
        df['installments_max'] = df['new_installments_max']+df['hist_installments_max']
        df['installments_ratio'] = df['new_installments_sum']/df['hist_installments_sum']
        df['price_total'] = df['purchase_amount_total'] / df['installments_total']
        df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
        df['price_max'] = df['purchase_amount_max'] / df['installments_max']
        df['duration_mean'] = df['new_duration_mean']+df['hist_duration_mean']
        df['duration_min'] = df['new_duration_min']+df['hist_duration_min']
        df['duration_max'] = df['new_duration_max']+df['hist_duration_max']
        df['amount_month_ratio_mean']=df['new_amount_month_ratio_mean']+df['hist_amount_month_ratio_mean']
        df['amount_month_ratio_min']=df['new_amount_month_ratio_min']+df['hist_amount_month_ratio_min']
        df['amount_month_ratio_max']=df['new_amount_month_ratio_max']+df['hist_amount_month_ratio_max']
        df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
        df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
        df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]

        train_df = train_df.drop(init_cols, axis=1)
        test_df = test_df.drop(init_cols, axis=1)

        self.train = train_df
        self.test = test_df

if __name__ == '__main__':
    generate_features(globals())
    features_json = {}
    features_path = cwd.replace(this_folder,'/features')
    df, _ = load_datasets(features_path)
    features_json.update({'features':list(df.columns)})
    with open('features.json', 'w') as f:
        json.dump(features_json, f, indent=4)
