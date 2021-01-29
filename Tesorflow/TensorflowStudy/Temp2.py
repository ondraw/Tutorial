import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import utils
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import joblib
import matplotlib.pyplot as plt

###########################################################################
#주제 이전날짜 7일의 Open Hight Low Close의 값으로 내일의 Close가를 예측하는 모델이다.
###########################################################################

##########데이터 로드
df = pd.read_csv('https://raw.githubusercontent.com/kairess/stock_crypto_price_prediction/master/dataset/eth.csv')

##########데이터 분석

print(df.head())
'''
   Unnamed: 0        Date  AveragePrice  ...          type  year  region
0           0  2015-12-27          1.33  ...  conventional  2015  Albany
1           1  2015-12-20          1.35  ...  conventional  2015  Albany
2           2  2015-12-13          0.93  ...  conventional  2015  Albany
3           3  2015-12-06          1.08  ...  conventional  2015  Albany
4           4  2015-11-29          1.28  ...  conventional  2015  Albany

[5 rows x 14 columns]
'''

print(df.info())
'''
'''

print(df.describe())
'''
         Unnamed: 0  AveragePrice  ...    XLarge Bags          year
count  18249.000000  18249.000000  ...   18249.000000  18249.000000
mean      24.232232      1.405978  ...    3106.426507   2016.147899
std       15.481045      0.402677  ...   17692.894652      0.939938
min        0.000000      0.440000  ...       0.000000   2015.000000
25%       10.000000      1.100000  ...       0.000000   2015.000000
50%       24.000000      1.370000  ...       0.000000   2016.000000
75%       38.000000      1.660000  ...     132.500000   2017.000000
max       52.000000      3.250000  ...  551693.650000   2018.000000

[8 rows x 11 columns]
'''

##########데이터 전처리

df = df[['Open', 'High', 'Low', 'Close']]
print(df)
'''
           Date  AveragePrice
2652 2015-12-27          0.95
2653 2015-12-20          0.98
2654 2015-12-13          0.93
2655 2015-12-06          0.89
2656 2015-11-29          0.99
        ...           ...
9097 2018-02-04          0.87
9098 2018-01-28          1.09
9099 2018-01-21          1.08
9100 2018-01-14          1.20
9101 2018-01-07          1.13
'''

data = df.values



train = data[:(len(data) - int(len(data)*0.3))]
test = data[:int(len(data)*0.3)]

transformer = MinMaxScaler()
train = transformer.fit_transform(train) #정규화(0~1)를 하기전에 표준편차를 하여 편차를 비슷하게 한후에 정규화(0~1)를 한다.
test = transformer.transform(test)

print('train ' , train[:5])
print('test ' , test[:5])
print('len ' , len(train) )

#window : 우리가 데이터를 바라보는 영역을 말한다. 0~window 길이만큼만 바라본다.
sequence_length = 7 #input 데이터는 7개의 값이 들어와야 한다. 즉 하나의 값으로 추측을 하지 않고 7개의 값을 묶어서 추측을 한다. 
window_length = sequence_length + 1

 

#train 데이터를 x,y로 분리 한다. x = 0~window_length, y = 마지막데이터. 
x_train = []
y_train = []
for i in range(0, len(train) - window_length + 1): #7Row단위로 데이터를 만들기 때문에 마지막 7로 전까지만 루프를 돈다.
    window = train[i:i + window_length, :] #0~7 row 즉 8개
    x_train.append(window[:-1, :]) #x~7 row 7개.
    y_train.append(window[-1, [-1]]) #8 row 의 마지막 열 (Close)
x_train = np.array(x_train)
y_train = np.array(y_train)

print('x_train shape=',x_train.shape,', y_train shape=',y_train.shape)


#test 데이터를 x,y로 분리 한다. x = 0~window_length, y = 마지막데이터.
x_test = []
y_test = []
for i in range(0, len(test) - window_length + 1): #7Row단위로 데이터를 만들기 때문에 마지막 7로 전까지만 루프를 돈다.
    window = test[i:i + window_length, :]  #x~7 row
    x_test.append(window[:-1, :])   #x~6 row
    y_test.append(window[-1, [-1]]) #7 row 의 마지막 열 (Close)
x_test = np.array(x_test)
y_test = np.array(y_test)


#섞어준다...
utils.shuffle(x_train, y_train)

#normalize한 정보값을 저장한다. (transformer를 저장하여 나중에 읽어들일때, 재사용한다.)
joblib.dump(transformer, './Work.ethereum_price_regression_model_transformer.pkl')

##########모델 학습
##########모델 검증

input = Input(shape=(sequence_length, 4))
net = LSTM(units=10)(input) 
net = Dense(units=1)(net)
model = Model(inputs=input, outputs=net)
model.summary()
'''
'''

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01))
#ModelCheckpoint 모델을 저장한다.
model.fit(x_train, y_train, epochs=60, validation_data=(x_test, y_test), callbacks=[ModelCheckpoint(filepath='./Work.ethereum_price_regression_model.h5', save_best_only=True, verbose=1)]) 

##########모델 예측
y_test_inverse = []
for y in y_test:
    inverse = transformer.inverse_transform([[0, 0, 0, y[0]]])
    y_inverse = inverse.flatten()[-1]
    print(y_inverse)
    y_test_inverse.append(y_inverse)

y_predict = model.predict(x_test)
y_predict_inverse = []
for y in y_predict:
    inverse = transformer.inverse_transform([[0, 0, 0, y[0]]])
    y_inverse = inverse.flatten()[-1]
    print(y_inverse)
    y_predict_inverse.append(y_inverse)


plt.plot(y_test_inverse)
plt.plot(y_predict_inverse)
plt.xlabel('Time Period')
plt.ylabel('Close')
plt.show()

x_test = transformer.transform([[307.38,310.55,305.88,305.88],
                                [305.76,306.40,290.58,291.69],
                                [290.73,293.91,281.17,287.43],
                                [288.50,308.31,287.69,305.71],
                                [305.48,305.48,295.80,300.47],
                                [300.04,301.37,295.12,296.26],
                                [296.43,305.42,293.72,298.89]])
x_test = x_test.reshape((1, 7, 4))
print(x_test)

y_predict = model.predict(x_test)
print(y_predict) #[[0.6553757]]
print(y_predict.flatten()[0]) #0.6553757

#normalize한 정보값으로 normalize된 값을 원래 값으로 변경한다.
inverse = transformer.inverse_transform([[0, 0, 0, y_predict.flatten()[0]]])
print(inverse.flatten()[-1])
