import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime

##########데이터 로드

data = pd.read_csv('https://raw.githubusercontent.com/kairess/stock_crypto_price_prediction/master/dataset/eth.csv')
print(data.head())

##########데이터 분석

##########데이터 전처리

high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices) / 2

seq_len = 50
sequence_length = seq_len + 1

#row의 개수를 50개로 만들고, 만들때 앞쪽 한자리씩 밀려서 만들어준다.
result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index: index + sequence_length])


#각 row의 시작 부분을 기준으로 nomalize을 했다. 한개의 픽셀단위로 이동한 것들이라 window[0]은 다음 필드에서 거의 비슷하다.
def normalize_windows(data):
    normalized_data = []
    for window in data:
        normalized_window = [1-((float(p) / float(window[0]))) for p in window]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)

result = normalize_windows(result)
print('데이터 프린트 ' , result[0,:10])

# split train and test data
row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1] #데이터의 마지막 전 열까지 


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #283,50,1 텐서로 변경.
y_train = train[:, -1] #데이터의 마지막을 Y로 설정한다.

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

x_train.shape, x_test.shape

##########모델 학습
##########모델 검증

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))

model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=10,
    epochs=20,
    callbacks=[
        TensorBoard(log_dir='Work.log/%s' % (start_time)),
        ModelCheckpoint('./Work.models/%s_eth.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
])

#model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[ModelCheckpoint(filepath='model/bit_coin_price_regression_model.h5', save_best_only=True, monitor='val_loss', mode='auto', verbose=1)])

##########모델 예측

pred = model.predict(x_test)

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()