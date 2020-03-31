import pandas as pd
import numpy as np
from scipy import optimize
from sklearn.metrics import f1_score
from sys import argv

"""
不平衡多份类阈值优化
要求：验证集和测试集的分布一致
"""
def get_f1(x, df):
    star = x[0]*df.star.values
    galaxy = x[1]*df.galaxy.values
    qso = x[2]*df.qso.values
    pred = np.vstack((star, galaxy, qso))
    pred = np.argmax(pred, axis=0)
    return -f1_score(df.label.values, pred, average='macro')


valid_result = pd.read_csv(argv[1])
print('修正前F1：', f1_score(valid_result.label.values,
                         np.argmax(valid_result[['star', 'galaxy', 'qso']].values, axis=1),
                         average='macro'))
x = np.array([1, 1, 1])
yz = optimize.fmin_powell(get_f1, x, args=(valid_result, ), maxiter=20)

test_result = pd.read_csv(argv[2])
test_result.star = yz[0]*test_result.star.values
test_result.galaxy = yz[1]*test_result.galaxy.values
test_result.qso = yz[2]*test_result.qso.values
test_result.label = np.argmax(test_result[['star', 'galaxy', 'qso']].values, axis=1)
test_result.label = test_result.label.map({0: 'star', 1: 'galaxy', 2 : 'qso' })
test_result[['id', 'label']].to_csv('test_xz.csv', index=False)


