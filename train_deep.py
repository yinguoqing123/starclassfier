import numpy as np
import os, glob
import pandas as pd
from keras.callbacks import Callback
from keras.utils import to_categorical
from tqdm import tqdm
from keras.layers import *
from keras.models import Model
import keras.backend as K
from sklearn.model_selection import train_test_split
import json
import gc


"""
初始分数：0.9816
1、流量归一化  0.9832
2、数据增强，改变batch内各个类的比例 0.9819
3、更改网络结构，去除两个滤波器相减操作 0.9812
4、更改网络结构，增加差分通道 
4、加入差分特征
"""

np.random.seed(2020)

def score_loss(y_true, y_pred):
    loss = 0
    for i in np.eye(3):
        y_true_ = K.constant([list(i)]) * y_true
        y_pred_ = K.constant([list(i)]) * y_pred
        loss += 0.5 * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
    return - K.log(loss + K.epsilon())

# 数据读取。
# 光谱的存储方式为一个txt存一个样本，txt里边是字符串格式的序列
# 对于做其他序列分类的读者，只需要知道这里就是生成序列的样本就就行了
class Data_Reader:
    def __init__(self, train_data, valid_data, features, features_diff):
        self.train_data = train_data
        self.valid_data = valid_data
        self.input_dim = 2600
        self.fracs = [0.7, 0.2, 0.1]  # 每个batch中，每一类的采样比例
        self.batch_size = 160  # batch_size
        for i in range(3):
            self.fracs[i] = int(self.fracs[i] * self.batch_size)
        self.fracs[0] = self.batch_size - np.sum(self.fracs[1:])
        self.features = features
        self.features_diff = features_diff
    def for_train(self):  # 生成训练集
        train_data = []
        for i in range(3):
            train_data.append(np.concatenate([np.expand_dims(self.train_data[self.train_data.label==i][self.features].values, 2),
                              np.expand_dims(self.train_data[self.train_data.label==i][self.features_diff].values, 2)], 2))
        del self.train_data
        gc.collect()
        Y = np.array([0] * self.fracs[0] + [1] * self.fracs[1] + [2] * self.fracs[2])
        Y = to_categorical(np.array(Y), 3)
        while True:
            X = []
            for i in range(3):
                for n in np.random.choice(len(train_data[i]), self.fracs[i]):
                    X.append(train_data[i][n])
            X = np.array(X)
            yield X, Y
            X = []
    def for_valid(self):  # 生成验证集
        cur = 0
        steps = (len(self.valid_data)+self.batch_size-1)//self.batch_size
        for i in range(steps):
            if i == steps-1:
                X = np.concatenate([np.expand_dims(self.valid_data.iloc[cur:][self.features].values, 2),
                              np.expand_dims(self.valid_data[cur:][self.features_diff].values, 2)], 2)
                Y = self.valid_data.iloc[cur:, :]['label'].values
                Y = to_categorical(np.array(Y), 3)
                yield X, Y
            else:
                X = np.concatenate([np.expand_dims(self.valid_data.iloc[cur:cur+self.batch_size][self.features].values, 2),
                                   np.expand_dims(self.valid_data[cur:cur+self.batch_size][self.features_diff].values, 2)], 2)
                Y = self.valid_data.iloc[cur:cur+self.batch_size, :]['label'].values
                Y = to_categorical(np.array(Y), 3)
                cur += self.batch_size
                yield X, Y

train_data = pd.read_pickle('train_data.pkl')
train_data.set_index('id', inplace=True)
train_data.rename(columns={'answer': 'label'}, inplace=True)
train_data.label = train_data.label.map({'star': 0, 'galaxy': 1, 'qso': 2})
features = train_data.columns.tolist()
features.remove('label')
features_diff = ['diff_'+feat for feat in features]

# 增加差分特征 流量归一化
def add_diff_channel(data, features):
    data_diff = data[features].diff(1, axis=1).add_prefix('diff_')
    data_diff.iloc[:, 0] = 0  # 第一列nan值填充为0
    data_diff.iloc[:] = data_diff.apply(lambda x: x / np.sqrt(sum(x ** 2)), axis=1)
    data[features] = data[features].apply(lambda x: x / np.sqrt(sum(x ** 2)), axis=1)
    data = data.merge(data_diff, left_index=True, right_index=True)
    del data_diff
    gc.collect()
    return data

train_data = add_diff_channel(train_data, features)

train_data, valid_data = train_test_split(train_data, test_size=0.2, stratify=train_data.label)

""" 
# 数据增强
def  data_aug(data, features, cls):
    tmp = data[data.label==cls].copy()
    tmp['variance'] = tmp[features].diff(1, axis=1).median(axis=1)
    tmp_aug = tmp[features + ['variance']].apply(lambda x: x + np.random.randn() * x.variance * 0.1, axis=1)[
        features]
    tmp_aug['label'] = cls
    return tmp_aug

galaxy_aug = data_aug(train_data, features, 1)
qso_aug_list = [data_aug(train_data, features, 2) for i in range(4)]
qso_aug = pd.concat(qso_aug_list)

train_data = pd.concat([train_data, galaxy_aug, qso_aug])
"""

D = Data_Reader(train_data, valid_data, features, features_diff)


def BLOCK(seq, filters):  # 定义网络的Block
    cnn = Conv1D(filters , 3, padding='SAME', dilation_rate=1, activation='relu')(seq)
    #cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
    cnn = Conv1D(filters , 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
    #cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
    cnn = Conv1D(filters , 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
    #cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(filters, 1, padding='SAME')(seq)
    seq = add([seq, cnn])
    return seq


# 搭建模型，就是常规的CNN加残差加池化
input_tensor = Input(shape=(D.input_dim, 2))
seq = input_tensor

seq = BLOCK(seq, 16)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 16)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 32)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 32)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 64)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 64)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 128)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 128)
seq = Dropout(0.5, (D.batch_size, int(seq.shape[1]), 1))(seq)
seq = GlobalMaxPooling1D()(seq)
seq = Dense(128, activation='relu')(seq)

output_tensor = Dense(3, activation='softmax')(seq)
model = Model(inputs=[input_tensor], outputs=[output_tensor])
model.summary()


# 定义marco f1 score的相反数作为loss
def score_loss(y_true, y_pred):
    loss = 0
    for i in np.eye(3):
        y_true_ = K.constant([list(i)]) * y_true
        y_pred_ = K.constant([list(i)]) * y_pred
        loss += (2/3) * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
    return - K.log(loss + K.epsilon())


# 定义marco f1 score的计算公式
def score_metric(y_true, y_pred):
    y_true = K.argmax(y_true)
    y_pred = K.argmax(y_pred)
    score = 0.
    for i in range(3):
        y_true_ = K.cast(K.equal(y_true, i), 'float32')
        y_pred_ = K.cast(K.equal(y_pred, i), 'float32')
        score += (2/3) * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
    return score


from keras.optimizers import Adam, SGD

model.compile(loss='categorical_crossentropy',  # 交叉熵作为loss
              optimizer=Adam(1e-3),
              metrics=[score_metric])

try:
    model.load_weights('guangpu_highest.model')
except:
    pass


def predict():
    import time
    cur = 0
    batch_size = 320
    test_data = pd.read_pickle('val_data.pkl')
    test_data.set_index('id', inplace=True)
    steps = (len(test_data) + batch_size - 1) // batch_size
    global features, features_diff
    test_data = add_diff_channel(test_data, features)
    Y = []
    proba = []
    for i in range(steps):
        if i == steps - 1:
            X = np.concatenate([np.expand_dims(test_data.iloc[cur:][features].values, 2),
                               np.expand_dims(test_data[cur:][features_diff].values, 2)], 2)
            y = model.predict(X)
            proba.extend(y)
            y = y.argmax(axis=1)
            Y.extend(list(y))
        else:
            X = np.concatenate([np.expand_dims(test_data.iloc[cur:cur+batch_size][features].values, 2),
                               np.expand_dims(test_data[cur:cur+batch_size][features_diff].values, 2)], 2)
            y = model.predict(X)
            proba.extend(y)
            y = y.argmax(axis=1)
            Y.extend(list(y))
            cur += batch_size
    d = pd.DataFrame({'id': test_data.index})
    proba = np.array(proba)
    d.loc[:, 'label'] = Y
    d['label'] = d['label'].map({0: 'star', 1: 'galaxy', 2: 'qso'})
    d.loc[:, 'star'] = proba[:, 0]
    d.loc[:, 'galaxy'] = proba[:, 1]
    d.loc[:, 'qso'] = proba[:, 2]
    d.to_csv('result_%s.csv' % (int(time.time())), index=None)

def get_bad_case(valid_data):
    import time
    cur = 0
    batch_size = 320
    steps = (len(valid_data) + batch_size - 1) // batch_size
    features = valid_data.columns.tolist()
    features.remove('label')
    Y = []
    proba = []
    for i in range(steps):
        if i == steps - 1:
            X = np.concatenate([np.expand_dims(valid_data.iloc[cur:][features].values, 2),
                                np.expand_dims(valid_data[cur:][features_diff].values, 2)], 2)
            y = model.predict(X)
            proba.extend(y)
            y = y.argmax(axis=1)
            Y.extend(list(y))
        else:
            X = np.concatenate([np.expand_dims(valid_data.iloc[cur:cur+batch_size][features].values, 2),
                                np.expand_dims(valid_data[cur:cur+batch_size][features_diff].values, 2)], 2)
            y = model.predict(X)
            proba.extend(y)
            y = y.argmax(axis=1)
            Y.extend(list(y))
            cur += batch_size
    d = pd.DataFrame({'id': valid_data.index})
    proba = np.array(proba)
    d.loc[:, 'star'] = proba[:, 0]
    d.loc[:, 'galaxy'] = proba[:, 1]
    d.loc[:, 'qso'] = proba[:, 2]
    d.loc[:, 'label'] = valid_data.label.values
    d.to_csv('result_proba_%s.csv' % (int(time.time())), index=None)



if __name__ == '__main__':
    # 定义Callback器，计算验证集的score，并保存最优模型
    class Evaluate(Callback):
        def __init__(self):
            self.scores = []
            self.highest = 0.
        def on_epoch_end(self, epoch, logs=None):
            R, T = [], []
            for x, y in D.for_valid():
                y_ = model.predict(x)
                R.extend(list(y.argmax(axis=1)))
                T.extend(list(y_.argmax(axis=1)))
            R, T = np.array(R), np.array(T)
            score = 0
            for i in range(3):
                R_ = (R == i)
                T_ = (T == i)
                precision = (R_*T_).sum()/(R_.sum() + K.epsilon())
                recall = (R_*T_).sum()/(T_.sum() + K.epsilon())
                f1_score = (2*precision*recall)/(precision+recall)
                score += (2/3) * (R_ * T_).sum() / (R_.sum() + T_.sum() + K.epsilon())
                print(f'{i}类Precision:{precision}, Recall:{recall}, F1:{f1_score}')
            self.scores.append(score)
            if score >= self.highest:  # 保存最优模型权重
                self.highest = score
                model.save_weights('guangpu_highest.model')
            json.dump([self.scores, self.highest], open('valid.log', 'w'))
            print('score: %f%%, highest: %f%%' % (score * 100, self.highest * 100))


    evaluator = Evaluate()

    # 第一阶段训练
    history = model.fit_generator(D.for_train(),
                                  steps_per_epoch=500,
                                  epochs=30,
                                  callbacks=[evaluator])

    model.compile(loss=score_loss,  # 换一个loss
                  optimizer=Adam(1e-4),
                  metrics=[score_metric])

    try:
        model.load_weights('guangpu_highest.model')
    except:
        pass

    # 第二阶段训练
    history = model.fit_generator(D.for_train(),
                                  steps_per_epoch=500,
                                  epochs=30,
                                  callbacks=[evaluator])
    try:
        model.load_weights('guangpu_highest.model')
    except:
        pass

    predict()
    # 保存验证集结果，分析bad case
    get_bad_case(valid_data)

