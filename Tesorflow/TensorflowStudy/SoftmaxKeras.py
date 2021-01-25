import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential # 케라스의 Sequential()을 임포트
from tensorflow.keras.layers import Dense # 케라스의 Dense()를 임포트
from tensorflow.keras import optimizers # 케라스의 옵티마이저를 임포트
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
from _cffi_backend import callback
import random

# x_test = np.array([[12,12,12],[35,36,34]])
# # 주의 : shape=[4] 이면 0~4의 값만 와야 한다. 
# y_test = np.array([0,3]) # sparse_softmax_cross_entropy_with_logits 을 사용하려면 logits형태의 값이다.

x_valid = np.array([[12,12,12],[35,36,38]])
# 주의 : shape=[4] 이면 0~4의 값만 와야 한다. 
y_valid = np.array([0,3])

# x_train = np.array([[10,10,10],
#                     [20,20,20],
#                     [30,30,30],
#                     [40,40,40],
#                     [11,11,11],
#                     [22,22,22],
#                     [33,33,33],
#                     [44,44,44],
#                     [12,12,12],
#                     [23,23,23],
#                     [34,34,34],
#                     [45,45,45]])
# 
# # 레이블 데이터에 one-hot encoding을 적용합니다.
# y_train = np.array([0,1,2,3,0,1,2,3,0,1,2,3])

arrX = []
arrY = []
for i in range(6000):
    index = random.randrange(0,4)
    value1 = (index + 1)*10 + random.randrange(0,10)
    value2 = (index + 1)*10 + random.randrange(0,10)
    value3 = (index + 1)*10 + random.randrange(0,10)
    arrX.append([value1,value2,value3])
    arrY.append(index)
    
    
x_train = np.array(arrX)
y_train = np.array(arrY)


arrX = []
arrY = []
for i in range(60):
    index = random.randrange(0,4)
    value1 = (index + 1)*10 + random.randrange(0,10)
    value2 = (index + 1)*10 + random.randrange(0,10)
    value3 = (index + 1)*10 + random.randrange(0,10)
    arrX.append([value1,value2,value3])
    arrY.append(index)
x_test = np.array(arrX)
y_test = np.array(arrY)

y_train, y_test, y_valid = to_categorical(y_train), to_categorical(y_test) , to_categorical(y_valid)


tensorBoard = TensorBoard(log_dir='./Work.log/{}'.format(time()))
tf.summary.trace_on(graph=True)
model=Sequential()
model.add(Dense(4, input_dim=3, activation='softmax'))
sgd=optimizers.SGD(lr=0.05)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# history = model.fit(x_train,y_train, batch_size=200, epochs=300, validation_data=(x_valid, y_valid),verbose=1,callbacks=[tensorBoard])
history = model.fit(x_train,y_train, batch_size=20, epochs=100, verbose=1,callbacks=[tensorBoard])
print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))

# writer = tf.summary.create_file_writer('./log')
# epochs = range(1, len(history.history['accuracy']) + 1)
# for i in epochs:
#     with writer.as_default():
#         tf.summary.scalar('loss',history.history['val_loss'][i],step=epochs)
    

model.summary()    
#keras.utils.plot_model(model, show_shapes=True)

epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'])
#plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


