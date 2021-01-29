import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
 
from sklearn import utils
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import matplotlib.pyplot as plt
 
###########################################################################
#주제 이전날짜 7일의 Open Hight Low Close의 값으로 내일의 Close가를 예측하는 모델이다.
###########################################################################
 
 
#날짜,컬럼개수,    볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격   볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격
col_Names=["time", "count", "v10", "p10","v9", "p9","v8", "p8","v7", "p7","v6", "p6","count2","v5", "p5","v4", "p4","v3", "p3","v2", "p2","v1", "p1"]
pd = pd.read_csv('/Users/songs/Downloads/MBMData/20210126/ETH.txt',names=col_Names)


#볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격   볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격
pd = pd[["v10", "p10","v9", "p9","v8", "p8","v7", "p7","v6", "p6","v5", "p5","v4", "p4","v3", "p3","v2", "p2","v1", "p1"]]
# pd = pd[["p10","p9","p8","p7","p6","p5","p4","p3","p2","p1"]]
pd = pd.drop_duplicates() #데이터량이 많기 때문에 중복 데이터를 제거 하자.
print(pd.info())

data = pd.values
train = data[:(len(data) - int(len(data)*0.1))]
test = data[:int(len(data)*0.1)]
  
transformer = MinMaxScaler()
train = transformer.fit_transform(train) #정규화(0~1)를 하기전에 표준편차를 하여 편차를 비슷하게 한후에 정규화(0~1)를 한다.
test = transformer.transform(test) #MinMax 정규화.

#window : 우리가 데이터를 바라보는 영역을 말한다. 0~window 길이만큼만 바라본다.
sequence_length = 60 #input 데이터는 7개의 값이 들어와야 한다. 즉 하나의 값으로 추측을 하지 않고 7개의 값을 묶어서 추측을 한다. 
window_length = sequence_length + 1
  
#train 데이터를 x,y로 분리 한다. x = 0~window_length, y = 마지막데이터. 
x_train = []
y_train = []
for i in range(0, len(train) - window_length*2): 
    window = train[i:i + window_length, :]
    x_train.append(window[:-1, :]) 
    y_train.append(train[i + window_length*2, [9]])
#     y_train.append(train[i + window_length*2, [4]]) 
      
x_train = np.array(x_train)
y_train = np.array(y_train)
  
print('x_train shape=',x_train.shape,', y_train shape=',y_train.shape)
x_test = []
y_test = []
for i in range(0, len(test) - window_length*2): 
    window = test[i:i + window_length, :]
    x_test.append(window[:-1, :]) 
    y_test.append(test[i + window_length*2, [9]])
#     y_test.append(test[i + window_length*2, [4]]) 
      
x_test = np.array(x_test)
y_test = np.array(y_test)



utils.shuffle(x_train, y_train)

joblib.dump(transformer, './Work.myeth_model_trans.pkl')
model = Sequential()
model.add(LSTM(sequence_length, return_sequences=True, input_shape=(sequence_length, 20)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))
# model.compile(loss='mse', optimizer='rmsprop')
model.compile(loss='mse', optimizer=Adam(lr=0.01))
model.summary()
model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), callbacks=[ModelCheckpoint(filepath='./Work.myeth_model.h5', save_best_only=True, verbose=1)])

 
 
##########모델 예측
#pd = pd[["v10", "p10","v9", "p9","v8", "p8","v7", "p7","v6", "p6","v5", "p5","v4", "p4","v3", "p3","v2", "p2","v1", "p1"]]
y_test_inverse = []
for y in y_test:
    inverse = transformer.inverse_transform([[0, 0,0, 0,0, 0,0, 0,0, y[0],0, 0,0, 0,0, 0,0, 0,0,0]])
#     inverse = transformer.inverse_transform([[0, 0,0, 0,y[0], 0,0, 0,0, 0]])
    y_inverse = inverse.flatten()[9]
    #print('y_inverse-------------\n',y_inverse)
    y_test_inverse.append(y_inverse)
 
y_predict = model.predict(x_test)
y_predict_inverse = []
for y in y_predict:
    inverse = transformer.inverse_transform([[0, 0,0, 0,0, 0,0, 0,0, y[0],0, 0,0, 0,0, 0,0, 0,0,0]])
#     inverse = transformer.inverse_transform([[0, 0,0, 0,y[0], 0,0, 0,0, 0]])
    y_inverse = inverse.flatten()[9]
    #print('y_inverse-------------\n',y_inverse)
    y_predict_inverse.append(y_inverse)
 
 
plt.plot(y_test_inverse,'ro')
plt.plot(y_predict_inverse,'bo')
plt.xlabel('Time Period')
plt.ylabel('Close')
plt.show()