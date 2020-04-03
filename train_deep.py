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
from keras.losses import  categorical_crossentropy

"""
初始分数：0.9816
1、流量归一化  有提升
2、数据增强，改变batch内各个类的比例 0.9819
3、更改网络结构，去除两个滤波器相减操作 0.9812
4、更改网络结构，增加差分通道 0.9829
5、将验证集中小样本加入训练 0.940 不知道是否存在过拟合风险，未在线上提交，max_f1优化后为0.962
6、将a榜数据加入训练  

只根据4提交了结果 线上0.9826
relu+softmax_entrypy： loss出现nan值，
原因：数据中出现了nan值
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
                              np.expand_dims(self.valid_data.iloc[cur:][self.features_diff].values, 2)], 2)
                Y = self.valid_data.iloc[cur:, :]['label'].values
                Y = to_categorical(np.array(Y), 3)
                yield X, Y
            else:
                X = np.concatenate([np.expand_dims(self.valid_data.iloc[cur:cur+self.batch_size][self.features].values, 2),
                                   np.expand_dims(self.valid_data.iloc[cur:cur+self.batch_size][self.features_diff].values, 2)], 2)
                Y = self.valid_data.iloc[cur:cur+self.batch_size, :]['label'].values
                Y = to_categorical(np.array(Y), 3)
                cur += self.batch_size
                yield X, Y


features = ['FE'+str(i) for i in range(2600)]
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

if not os.path.exists('train_data_normalize.pkl'):
    train_data = pd.read_pickle('train_data.pkl')
    train_data.set_index('id', inplace=True)
    train_data.rename(columns={'answer': 'label'}, inplace=True)
    train_data.label = train_data.label.map({'star': 0, 'galaxy': 1, 'qso': 2})
    train_data = add_diff_channel(train_data, features)
    train_data.to_pickle('train_data_normalize.pkl')
else:
    train_data = pd.read_pickle('train_data_normalize.pkl')
# 这些样本2600维全为1， 出现nan值
id_list = ['01703e6377847e1232e466ac18c3b77f', '03a7a75a6d27b27a2a4beeb1ef9f83a9',
       '0cd6b854b430974a9cff3caf1fa780ca', '0e44f3c097a1a12f614f4bdb4b10bdf1',
       '1082efc11732329f933752a810f9638e', '15e50a93629de58527474647f7577953',
       '1605d57fe264730b6b7bc0302e851907', '17b3a1e3dbb215d8129e22c9251aafb9',
       '1875e0da165004aaf9a3843a2c605644', '1a84416eb6f05016513f3a6df412c6d5',
       '1c29b2fc6b9c2a955d371e07aa4b21ba', '1e364be5461ee97296e30cc5f675d865',
       '221e67a9b1a09ab5faff9fe20244672e', '2a565745a52fbc5a36d658ee6d3ef4ca',
       '2e6c37e4aeb15e3ef71f86db0f854bbd', '2e8eacab4f4ee1fe286c8cdfb2d0b9e2',
       '2fbc59a095a43578c3c48b6f10a32fa9', '320396000f93c06ad6252022f5c823ed',
       '35cd808100d9627a457d694aca45cb9b', '3ba9295fcdb66dec02b926c76728f586',
       '3e297f5727bc1b0f927781c6f26796cf', '3ff32f148d36b8c7b7716cae70e458a4',
       '40b4475ef67ccb6be495ba7e7306a73c', '432e9277f02cb50cbfead4a598690368',
       '46d91b93560df1e530bb1f23aaa186e8', '474f687e9fe98b8dc5c6503b45839228',
       '47a70662ac5bcd29d13b0231f9f7104e', '4ed98d8e31794369688ba2b3292def0f',
       '58811ed9f45371624bd55c0e8af0cc03', '606ec43bcf40625164aec62543afcf18',
       '61e0ac120022d19e810e524a20f5d59a', '635a597ce0400a43000bfc2eaaf50509',
       '636588680f95da1219d81fc3147c4084', '64541a18375c93d05dc2144fa089a730',
       '6614e1477c5c27139521e16612a8b96c', '686e70684ea2681ec013aecd74dc0ae0',
       '6b963246340127b13783ac1ea4b8e910', '6e7962cac7b39cbb0d0bb547a641269f',
       '6ffcba65a50c7e3d69810ae0bb412c50', '767805f72365523192916f1ba6bb45b4',
       '7890916cc420a5a6444dba66bca3f045', '789c28ef83bf521ce797b27cc8ad85e9',
       '80f3fe532ed0ba20436386668444b2c6', '81725be21308ff789a2e5e1b59884260',
       '81e1178a6c78c24cb3d1e802cab7c32f', '890e03cc4fea98f90f3a0ce93c3c03f1',
       '8ad6a7cdba22246da289c33a3b8d0c15', '8da1301aaa26e95e77821d6868f1dca3',
       '8e36e95638b867ff9ed600302feb6904', '917d75ba8bcddfbabc12b11dca6041e8',
       '94f0aaca98802d98665e5923aa79f710', '96369a0d73b5f9cc4f42a774597eb65c',
       '97151a86109c97d16baacd12d6b248df', '986020efe3b290bea51d9521f5997eb6',
       '998ebc3e154b577e345099c71e939db6', '9bb3241fb93b387f2f5ddce36298e824',
       '9cddd8c1b56052fa1e29fb0985d2901c', 'a13ac2947b713bc6c8f41db03c30e94f',
       'a4b17db8c84b5bfe613bcacde8428103', 'a680a2b562a1616106292c181954da6d',
       'a817d7151550beeb27582a95624d9b67', 'add268adc1d753045051b764f5ae4360',
       'b1c5b691e65542339f89c97f579ae08a', 'b241f3185aa227dc285f10e907faf915',
       'b4a6ba6d51fb126398cad8b932a8ac46', 'b8a536e26b61547587d1b65034cf7ba3',
       'bd94f2755576405cd5e683cf84aea497', 'c24cbc0bef278cb8b4b9413597dc2783',
       'c29723e9a391cf30dbdfa3472875fba7', 'c3c6f01ec02a569ee1a52dd10b48adc3',
       'cb7d269674a2752a27ef6b6cff5b6a47', 'ce928dbbbc6353fc0dcc71b1d829e43d',
       'ce9cd15916aa40ff702c1af344d95ded', 'cff2601b4dc16aa883f723a482a027b7',
       'db091771789021300452c8fa3e721a7f', 'dda336d608e764cc717136f5f34c0957',
       'e318fb7933c001aaca2d68f5caa13dc8', 'e4540e35cee849e7d25595c7e5d86ebe',
       'ea18f3abab3bb6a717693ab1e3391363', 'ea81609d6802f7097893d6be81548cc0',
       'ee8957893397c0a4f580457ebab68d4d', 'f87de7c8882124b456bac216b2b46b29',
       'ffc69e40a69f8aeba3f761329cdc3e6a']

train_data = train_data[~train_data.index.isin(id_list)]
train_data, valid_data = train_test_split(train_data, test_size=0.2, stratify=train_data.label, random_state=2020)

# 训练集中加入valid_data中的galaxy和qso样本
#train_data = pd.concat([train_data, valid_data.loc[valid_data.label!=0]])

# 训练集中加入线上预测结果
add_offline_label = pd.read_csv('result_proba_1585479113.csv')
add_offline_label = add_offline_label[['id', 'label']]

if os.path.exists('val_data_normalize.pkl'):
    add_offline = pd.read_pickle('val_data_normalize.pkl')
else:
    add_offline = pd.read_pickle('val_data.pkl')
    add_offline.set_index('id', inplace=True)
    add_offline = add_diff_channel(add_offline, features)
    add_offline.to_pickle('val_data_normalize.pkl')

add_offline = add_offline.merge(add_offline_label, right_on='id', left_index=True)
train_data = pd.concat([train_data, add_offline[add_offline.label!=0]])

#数据增强
def  data_aug(data, features, cls):
    tmp = data[data.label==cls].copy()
    tmp['variance'] = tmp[features].diff(1, axis=1).median(axis=1)
    tmp_aug = tmp[features + ['variance']].apply(lambda x: x + np.random.randn() * x.variance * 0.1, axis=1)[
        features]
    tmp_aug['label'] = cls
    return tmp_aug

"""
galaxy_aug = data_aug(train_data, features, 1)
qso_aug_list = [data_aug(train_data, features, 2) for i in range(4)]
qso_aug = pd.concat(qso_aug_list)

train_data = pd.concat([train_data, galaxy_aug, qso_aug])
"""

D = Data_Reader(train_data, valid_data, features, features_diff)

def BLOCK(seq, filters, n1, n2=0):  # 定义网络的Block
    # 分组卷积
    batch_size, steps, channels = K.int_shape(seq)
    if n1 <=n2:
        seq_0 = Lambda(lambda seq: seq[:, :, :channels//2])(seq)
        seq_diff = Lambda(lambda seq: seq[:, :, channels//2:])(seq)
        cnn_diff = Conv1D(filters, 3, padding='SAME', dilation_rate=1, activation='relu')(seq_diff)
        # cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
        cnn_diff = Conv1D(filters, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn_diff)
        # cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
        cnn_diff = Conv1D(filters, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn_diff)
        cnn = Conv1D(filters , 3, padding='SAME', dilation_rate=1, activation='relu')(seq_0)
        #cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
        cnn = Conv1D(filters , 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
        #cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
        cnn = Conv1D(filters , 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
        #cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
        if n1 == n2:
            cnn = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=2))([cnn, cnn_diff])
            cnn = Conv1D(filters, 1, padding='SAME', activation='relu')(cnn)
            if channels != filters:
                seq = Conv1D(filters, 1, padding='SAME')(seq)
            seq = add([seq, cnn])
        else:
            if channels != filters:
                seq_0 = Conv1D(filters, 1, padding='SAME')(seq_0)
                seq_diff = Conv1D(filters, 1, padding='SAME')(seq_diff)
            seq_0 = add([seq_0, cnn])
            seq_diff = add([seq_diff, cnn_diff])
            seq = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=2))([seq_0, seq_diff])
    else:
        cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=1, activation='relu')(seq)
        # cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
        cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
        # cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
        cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
        cnn = Conv1D(filters, 3, padding='SAME', dilation_rate=6, activation='relu')(cnn)
        if int(seq.shape[-1]) != filters:
            seq = Conv1D(filters, 1, padding='SAME')(seq)
        seq = add([seq, cnn])
    return seq


# 搭建模型，就是常规的CNN加残差加池化
input_tensor = Input(shape=(D.input_dim, 2))
seq = input_tensor

seq = BLOCK(seq, 16, 1)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 16, 2)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 32, 3)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 32, 4)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 64, 5)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 64, 6)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 128, 7)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 128, 8)
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


def predict(datafile='val_data.pkl'):
    import time
    cur = 0
    batch_size = 320
    global features, features_diff
    if os.path.exists('{}_normalize.pkl'.format(datafile[:8])):
        test_data = pd.read_pickle('{}_normalize.pkl'.format(datafile[:8]))
    else:
        test_data = pd.read_pickle(datafile)
        test_data.set_index('id', inplace=True)
        test_data = add_diff_channel(test_data, features)
        test_data.to_pickle('{}_normalize.pkl'.format(datafile[:8]))
    steps = (len(test_data) + batch_size - 1) // batch_size
    Y = []
    proba = []
    for i in range(steps):
        if i == steps - 1:
            X = np.concatenate([np.expand_dims(test_data.iloc[cur:][features].values, 2),
                               np.expand_dims(test_data.iloc[cur:][features_diff].values, 2)], 2)
            y = model.predict(X)
            proba.extend(y)
            y = y.argmax(axis=1)
            Y.extend(list(y))
        else:
            X = np.concatenate([np.expand_dims(test_data.iloc[cur:cur+batch_size][features].values, 2),
                               np.expand_dims(test_data.iloc[cur:cur+batch_size][features_diff].values, 2)], 2)
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
    global features, features_diff
    Y = []
    proba = []
    for i in range(steps):
        if i == steps - 1:
            X = np.concatenate([np.expand_dims(valid_data.iloc[cur:][features].values, 2),
                                np.expand_dims(valid_data.iloc[cur:][features_diff].values, 2)], 2)
            y = model.predict(X)
            proba.extend(y)
            y = y.argmax(axis=1)
            Y.extend(list(y))
        else:
            X = np.concatenate([np.expand_dims(valid_data.iloc[cur:cur+batch_size][features].values, 2),
                                np.expand_dims(valid_data.iloc[cur:cur+batch_size][features_diff].values, 2)], 2)
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

    predict(datafile='test_data.pkl')
    # 保存验证集结果，分析bad case
    get_bad_case(valid_data)

