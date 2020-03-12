# -*- encoding: utf-8 -*-
'''
树模型训练
'''
import gc
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import time
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from bayes_opt import BayesianOptimization
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--search", help="search hyper parameters or not")
args = parser.parse_args()

train_data = pd.read_pickle('train_feature.pkl')
test_data = pd.read_pickle('val_feature.pkl')

features = train_data.columns.tolist()
features.remove('id')
features.remove('label')


# 取1/5的数据用于调参
if args.search:
    _, train_data = train_test_split(train_data, test_size=0.2, stratify=train_data.label)

train_set, valid_set = train_test_split(train_data, test_size=0.2, stratify=train_data.label)
print('train set 样本量：', train_set.shape[0])

#train_set.loc[:, features] = train_set[features].astype(np.float32)
#valid_set.loc[:, features] = valid_set[features].astype(np.float32)

def lgb_eval(num_leaves,  max_depth, lambda_l2,lambda_l1, min_child_samples, bagging_fraction,
             feature_fraction, min_child_weight):
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "lambda_l2": lambda_l2,
        "lambda_l1": lambda_l1,
        "num_threads": 32,
        "min_child_samples": int(min_child_samples),
        "min_child_weight": min_child_weight,
        "learning_rate": 0.05,
        "bagging_fraction": bagging_fraction,
        "feature_fraction": feature_fraction,
        "seed": 2020,
        "verbosity": -1
    }
    train_df = lgb.Dataset(train_set[features], train_set.label)
    scores = lgb.cv(params, train_df, num_boost_round=1000, early_stopping_rounds=30, verbose_eval=False,
                     nfold=3)['multi_logloss-mean'][-1]
    return scores

def param_tuning(init_points,num_iter,**args):
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (10, 120),
                                                'max_depth': (5, 15),
                                                'lambda_l2': (0.0, 3),
                                                'lambda_l1': (0.0, 3),
                                                'bagging_fraction': (0.5, 0.8),
                                                'feature_fraction': (0.3, 0.8),
                                                'min_child_samples': (20, 100),
                                                'min_child_weight': (0, 15)
                                                })

    lgbBO.maximize(init_points=init_points, n_iter=num_iter,**args)
    return lgbBO

if  args.search=='random':
    param_dict = {'n_estimators': range(300, 1000, 50), 'num_leaves': range(15, 50, 5),'max_depth': range(5, 10, 1),
                  'learning_rate': np.arange(0.05, 0.3, 0.05), 'subsample': np.arange(0.6, 1.0, 0.1),
                  'colsample_bytree': np.arange(0.3, 1.0, 0.1), 'min_child_weight': range(0, 10, 1),
                  'reg_alpha': np.arange(0.0, 2, 0.1), 'reg_lambda': np.arange(0.0, 1, 0.1)}
    clf= lgb.LGBMClassifier()
    fit_rounds = 300
    grid = RandomizedSearchCV(clf, param_dict, cv=3, scoring='neg_log_loss', n_iter=fit_rounds, n_jobs=4)
    fit_begin = time.time()
    grid.fit(train_set[features], train_set.label)
    model_lgb = grid.best_estimator_
    fit_end = time.time()
    print("参数搜索轮数：{}，总训练时间{}分钟".format(fit_rounds, (fit_end - fit_begin) / 60))
    test_data['label'] = model_lgb.predict(test_data[features])
    print("模型最佳参数：")
    print(grid.best_params_)
    test_data[['id', 'label']].to_csv('test_answer.csv', index=False)
elif args.search=='beyas':
    result = param_tuning(5, 40)
    print('模型最佳参数AUC:{}'.format(result.max['target']))
    params = result.max['params']
    print('模型最佳参数:', '\n', params)
    result.probe(params)
    result.maximize(0, 10)
    params = result.max['params']
    print('再次精调后最佳AUC:{}'.format(result.max['target']))
    print('再次精调后最佳参数：', '\n', result.max['params'])
else:
    model_lgb = lgb.LGBMClassifier(n_estimators=950, max_depth=9, num_leaves=15, n_jobs=-1, learning_rate=0.05,
                                   colsample_bytree=0.3, subsample=0.6, reg_lambda=0.9, reg_alpha=1.3,
                                   min_child_weight=5,
                                   random_state=2019)
    model_lgb.fit(train_set[features], train_set.label, eval_set=[(valid_set[features], valid_set.label)],
                  early_stopping_rounds=30)
    test_data['label'] = model_lgb.predict(test_data[features])
    test_data[['id', 'label']].to_csv('test_answer.csv', index=False)