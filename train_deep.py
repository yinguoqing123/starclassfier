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
"""
初始分数：0.98
1、流量归一化
2、数据增强
3、加入BN
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
    def __init__(self, train_data, valid_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.input_dim = self.train_data.shape[1] - 1
        self.fracs = [0.8, 0.1, 0.05]  # 每个batch中，每一类的采样比例
        self.batch_size = 160  # batch_size
        for i in range(3):
            self.fracs[i] = int(self.fracs[i] * self.batch_size)
        self.fracs[0] = self.batch_size - np.sum(self.fracs[1:])
        self.features = self.train_data.columns.tolist()
        self.features.remove('label')
    def for_train(self):  # 生成训练集
        train_data = []
        for i in range(3):
            train_data.append(self.train_data[self.train_data.label==i][self.features].values)
        Y = np.array([0] * self.fracs[0] + [1] * self.fracs[1] + [2] * self.fracs[2])
        Y = to_categorical(np.array(Y), 3)
        while True:
            X = []
            for i in range(3):
                for n in np.random.choice(len(train_data[i]), self.fracs[i]):
                    X.append(train_data[i][n])
            X = np.expand_dims(np.array(X), 2)
            yield X, Y
            X = []
    def for_valid(self):  # 生成验证集
        cur = 0
        steps = (len(self.valid_data)+self.batch_size-1)//self.batch_size
        for i in range(steps):
            if i == steps-1:
                X = self.valid_data.iloc[cur:, :][self.features].values
                X = np.expand_dims(np.array(X), 2)
                Y = self.valid_data.iloc[cur:, :]['label'].values
                Y = to_categorical(np.array(Y), 3)
                yield X, Y
            else:
                X = self.valid_data.iloc[cur:cur + self.batch_size, :][self.features].values
                X = np.expand_dims(np.array(X), 2)
                Y = self.valid_data.iloc[cur:cur + self.batch_size, :]['label'].values
                Y = to_categorical(np.array(Y), 3)
                cur += self.batch_size
                yield X, Y


train_data = pd.read_pickle('train_data.pkl')
train_data.drop(columns=['id'], inplace=True)
train_data.rename(columns={'answer': 'label'}, inplace=True)
train_data.label = train_data.label.map({'star': 0, 'galaxy': 1, 'qso': 2})
features = train_data.columns.tolist()
features.remove('label')
train_data.iloc[:][features] = train_data[features].apply(lambda x: x/np.sqrt(sum(x**2)), axis=1)
train_data, valid_data = train_test_split(train_data, test_size=0.2)


D = Data_Reader(train_data, valid_data)



def BLOCK(seq, filters):  # 定义网络的Block
    cnn = Conv1D(filters * 2, 3, padding='SAME', dilation_rate=1, activation='relu')(seq)
    cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
    cnn = Conv1D(filters * 2, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
    cnn = Conv1D(filters * 2, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(filters, 1, padding='SAME')(seq)
    seq = add([seq, cnn])
    return seq


# 搭建模型，就是常规的CNN加残差加池化
input_tensor = Input(shape=(D.input_dim, 1))
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
    test_data= pd.read_pickle('val_data.pkl')
    cur = 0
    batch_size = 160
    steps = (len(test_data) + batch_size - 1) // batch_size
    features = test_data.columns.tolist()
    features.remove('id')
    test_data.iloc[:][features] = test_data[features].apply(lambda x: x / np.sqrt(sum(x ** 2)), axis=1)
    Y = []
    for i in range(steps):
        if i == steps - 1:
            X = test_data.iloc[cur:, :][features].values
            X = np.expand_dims(np.array(X), 2)
            y = model.predict(X)
            y = y.argmax(axis=1)
            Y.extend(list(y))
        else:
            X = test_data.iloc[cur:cur + batch_size, :][features].values
            X = np.expand_dims(np.array(X), 2)
            y = model.predict(X)
            y = y.argmax(axis=1)
            Y.extend(list(y))
            cur += batch_size
    d = pd.DataFrame({'id': test_data.id})
    d.loc[:, 'label'] = Y
    d['label'] = d['label'].map({0: 'star', 1: 'galaxy', 2: 'qso'})
    d.to_csv('result_%s.csv' % (int(time.time())), index=None)


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
                                  epochs=20,
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
                                  epochs=20,
                                  callbacks=[evaluator])
    predict()

