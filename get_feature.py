import os
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook, tnrange
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pickle
import warnings
import time
warnings.filterwarnings("ignore")


def get_moving_windows_features(data, flag='train'):
    data_ = pd.DataFrame({'id': data.index})
    data_diff = data.diff(1, axis=1).drop(columns=data.columns[0])
    data_big0 = data > 0
    data_abs = data.apply(np.abs)
    data_abs_big0 = data_big0 * data_abs
    window_sizes = [100, 200, 500, 1300, 2600]
    for window_size in window_sizes:
        print('window size is ', window_size)
        len_ = int(data.shape[1]/window_size)
        mean_cols = []
        var_cols = []
        min_cols = []
        max_cols = []
        kurt_cols = []
        skew_cols = []
        sum_cols = []
        median_cols = []

        diffmean_cols = []
        diffvar_cols = []
        diffmin_cols = []
        diffmax_cols = []
        diffkurt_cols = []
        diffskew_cols = []
        diffsum_cols = []
        diffmedian_cols = []

        second_cols = {}

        for i in range(len_):
            tmp = data.iloc[:, i * window_size:(i + 1) * window_size].copy()
            tmp_diff = data_diff.iloc[:, i * window_size:(i + 1) * window_size].copy()
            tmp_big0 = data_big0.iloc[:, i * window_size:(i + 1) * window_size].copy()
            tmp_abs = data_abs.iloc[:, i * window_size:(i + 1) * window_size].copy()
            tmp_abs_big0 = data_abs_big0.iloc[:, i * window_size:(i + 1) * window_size].copy()

            data_[str(i) + '_' + str(window_size) + '_mean'] = tmp.mean(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_max'] = tmp.max(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_min'] = tmp.min(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_var'] = tmp.var(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_median'] = tmp.median(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_sum'] = tmp.sum(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_skew'] = tmp.skew(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_kurt'] = tmp.kurt(axis=1).astype(np.float32).values

            data_[str(i) + '_' + str(window_size) + '_range'] = data_[str(i) + '_' + str(window_size) + '_max'] - data_[
                str(i) + '_' + str(window_size) + '_min']
            data_[str(i) + '_' + str(window_size) + '_argmax'] = tmp.idxmax(axis=1).values
            data_[str(i) + '_' + str(window_size) + '_argmax'] = data_[str(i) + '_' + str(window_size) + '_argmax']\
                .apply(lambda x: str(x)[2:]).astype(np.int32)
            data_[str(i) + '_' + str(window_size) + '_argmin'] = tmp.idxmin(axis=1).values
            data_[str(i) + '_' + str(window_size) + '_argmin'] = data_[str(i) + '_' + str(window_size) + '_argmin']\
                .apply(lambda x:str(x)[2:]).apply(np.int32)

            # 一阶差分统计
            data_[str(i) + '_' + str(window_size) + '_diffmean'] = tmp_diff.mean(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_diffmax'] = tmp_diff.max(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_diffmin'] = tmp_diff.min(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_diffvar'] = tmp_diff.var(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_diffmedian'] = tmp_diff.median(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_diffsum'] = tmp_diff.sum(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_diffskew'] = tmp_diff.skew(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_diffkurt'] = tmp_diff.kurt(axis=1).astype(np.float32).values

            data_[str(i) + '_' + str(window_size) + '_diffrange'] = data_[str(i) + '_' + str(window_size) + '_diffmax'] \
                                                                    - data_[str(i) + '_' + str(window_size) + '_diffmin']
            data_[str(i) + '_' + str(window_size) + '_diffargmax'] = tmp_diff.idxmax(axis=1).values
            data_[str(i) + '_' + str(window_size) + '_diffargmax'] = data_[str(i) + '_' + str(window_size) + '_diffargmax']\
                .apply(lambda x:str(x)[2:]).astype(np.int32)
            data_[str(i) + '_' + str(window_size) + '_diffargmin'] = tmp_diff.idxmin(axis=1).values
            data_[str(i) + '_' + str(window_size) + '_diffargmin'] = data_[str(i) + '_' + str(window_size) + '_diffargmin']\
                .apply(lambda x:str(x)[2:]).astype(np.int32)
            # 大于0的个数
            data_[str(i) + '_' + str(window_size) + '_abovezero'] = tmp_big0.sum(axis=1).astype(np.float32).values

            # abs
            data_[str(i) + '_' + str(window_size) + '_absmean'] = tmp_abs.mean(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_absmax'] = tmp_abs.max(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_absmin'] = tmp_abs.min(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_absvar'] = tmp_abs.var(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_absmedian'] = tmp_abs.median(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_abssum'] = tmp_abs.sum(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_absskew'] = tmp_abs.skew(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_abskurt'] = tmp_abs.kurt(axis=1).astype(np.float32).values

            # abs_big0
            data_[str(i) + '_' + str(window_size) + '_abs_big0_mean'] = tmp_abs_big0.mean(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_abs_big0_max'] = tmp_abs_big0.max(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_abs_big0_min'] = tmp_abs_big0.min(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_abs_big0_var'] = tmp_abs_big0.var(axis=1).astype(np.float32).values
            # data_[str(i)+'_'+str(window_size)+'_abs_big0_median'] = tmp_abs_big0.median(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_abs_big0_skew'] = tmp_abs_big0.skew(axis=1).astype(np.float32).values
            data_[str(i) + '_' + str(window_size) + '_abs_big0_kurt'] = tmp_abs_big0.kurt(axis=1).astype(np.float32).values

            mean_cols.append(str(i) + '_' + str(window_size) + '_mean')
            max_cols.append(str(i) + '_' + str(window_size) + '_max')
            min_cols.append(str(i) + '_' + str(window_size) + '_min')
            var_cols.append(str(i) + '_' + str(window_size) + '_var')
            median_cols.append(str(i) + '_' + str(window_size) + '_median')
            sum_cols.append(str(i) + '_' + str(window_size) + '_sum')
            skew_cols.append(str(i) + '_' + str(window_size) + '_skew')
            kurt_cols.append(str(i) + '_' + str(window_size) + '_kurt')

            diffmean_cols.append(str(i) + '_' + str(window_size) + '_diffmean')
            diffmax_cols.append(str(i) + '_' + str(window_size) + '_diffmax')
            diffmin_cols.append(str(i) + '_' + str(window_size) + '_diffmin')
            diffvar_cols.append(str(i) + '_' + str(window_size) + '_diffvar')
            # diffmedian_cols.append(str(i)+'_'+str(window_size)+'_diffmedian')
            # diffsum_cols.append(str(i)+'_'+str(window_size)+'_diffsum')
            diffskew_cols.append(str(i) + '_' + str(window_size) + '_diffskew')
            diffkurt_cols.append(str(i) + '_' + str(window_size) + '_diffkurt')

        if window_size > 1000:
            continue

        # 对序列所有区间的统计值再做统计值
        second_cols[str(window_size) + '_mean'] = mean_cols
        second_cols[str(window_size) + '_max'] = max_cols
        second_cols[str(window_size) + '_min'] = min_cols
        second_cols[str(window_size) + '_var'] = var_cols
        second_cols[str(window_size) + '_median'] = median_cols
        second_cols[str(window_size) + '_sum'] = sum_cols
        second_cols[str(window_size) + '_skew'] = skew_cols
        second_cols[str(window_size) + '_kurt'] = kurt_cols

        second_cols[str(window_size) + '_diffmean'] = diffmean_cols
        second_cols[str(window_size) + '_diffmax'] = diffmax_cols
        second_cols[str(window_size) + '_diffmin'] = diffmin_cols
        second_cols[str(window_size) + '_diffvar'] = diffvar_cols
        # second_cols[str(window_size)+'_diffmedian'] = diffmedian_cols
        # second_cols[str(window_size)+'_diffsum'] = diffsum_cols
        second_cols[str(window_size) + '_diffskew'] = diffskew_cols
        second_cols[str(window_size) + '_diffkurt'] = diffkurt_cols

        for key in second_cols.keys():
            cols = second_cols[key]

            data_[key + '_mean'] = data_[cols].mean(axis=1).astype(np.float32).values
            data_[key + '_max'] = data_[cols].max(axis=1).astype(np.float32).values
            data_[key + '_min'] = data_[cols].min(axis=1).astype(np.float32).values
            data_[key + '_var'] = data_[cols].var(axis=1).astype(np.float32).values
            data_[key + '_median'] = data_[cols].median(axis=1).astype(np.float32).values
            data_[key + '_sum'] = data_[cols].sum(axis=1).astype(np.float32).values
            data_[key + '_skew'] = data_[cols].skew(axis=1).astype(np.float32).values
            data_[key + '_kurt'] = data_[cols].kurt(axis=1).astype(np.float32).values

            data_[key + '_diffmean'] = data_[cols].diff(1, axis=1).mean(axis=1).astype(np.float32).values
            data_[key + '_diffmax'] = data_[cols].diff(1, axis=1).max(axis=1).astype(np.float32).values
            data_[key + '_diffmin'] = data_[cols].diff(1, axis=1).min(axis=1).astype(np.float32).values
            data_[key + '_diffvar'] = data_[cols].diff(1, axis=1).var(axis=1).astype(np.float32).values
            data_[key + '_diffskew'] = data_[cols].diff(1, axis=1).skew(axis=1).astype(np.float32).values
            data_[key + '_diffkurt'] = data_[cols].diff(1, axis=1).kurt(axis=1).astype(np.float32).values

    data_.to_pickle(f'{flag}_feature.pkl')



train_dtypes = {}
val_dtypes = {}
for fea in ['FE' + str(i) for i in range(2600)]:
    train_dtypes[fea] = np.float32
    val_dtypes[fea] = np.float32

train_dtypes['answer'] = np.str
train_dtypes['id'] = np.str
val_dtypes['id'] = np.str
train_data = pd.read_pickle('train_data.pkl')
train_data.answer = train_data.answer.map({'star': 0, 'galaxy': 1, 'qso': 2})
print('train data memory before:', train_data.info(memory_usage='deep'))
train_data = train_data.astype(train_dtypes, copy=False)
print('train data memory after:', train_data.info(memory_usage='deep'))

#数据集不均衡  start:galaxy:qso = 0.839 : 0.122: 0.038

val_data = pd.read_pickle('val_data.pkl')
print('test data memory before:', val_data.info(memory_usage='deep'))
val_data = val_data.astype(val_dtypes, copy=False)
print('test data memory after:', val_data.info(memory_usage='deep'))

start_time = time.time()
train_data = train_data.set_index('id').drop(columns='answer')
get_moving_windows_features(train_data)
train_end_time = time.time()
print('训练集处理时间为{}分钟'.format((train_end_time-start_time)/60))
del train_data
gc.collect()
val_data = val_data.set_index('id')
get_moving_windows_features(val_data, flag='val')
print('验证集处理时间为{}分钟'.format((time.time()-train_end_time)/60))

