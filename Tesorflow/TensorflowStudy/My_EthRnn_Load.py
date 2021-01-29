import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
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
pd = pd.read_csv('/Users/songs/Downloads/MBMData/20210125/ETH.txt',names=col_Names)

#볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격   볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격,볼륨,가격
pd = pd[["v10", "p10","v9", "p9","v8", "p8","v7", "p7","v6", "p6","v5", "p5","v4", "p4","v3", "p3","v2", "p2","v1", "p1"]]
pd = pd.drop_duplicates() #데이터량이 많기 때문에 중복 데이터를 제거 하자.
print(pd.info())

test = pd.values
test = test[:200,:]
  
transformer = joblib.load('./Work.myeth_model_trans.pkl') 
test = transformer.transform(test) #MinMax 정규화.
model = load_model('./Work.myeth_model.h5')


#window : 우리가 데이터를 바라보는 영역을 말한다. 0~window 길이만큼만 바라본다.
sequence_length = 60 #input 데이터는 7개의 값이 들어와야 한다. 즉 하나의 값으로 추측을 하지 않고 7개의 값을 묶어서 추측을 한다. 
window_length = sequence_length + 1

x_test = []
y_test = []
for i in range(0, len(test) - window_length*2): 
    window = test[i:i + window_length, :]
    x_test.append(window[:-1, :]) 
    y_test.append(test[i + window_length*2, [9]])
      
x_test = np.array(x_test)
y_test = np.array(y_test)

 
##########모델 예측
#pd = pd[["v10", "p10","v9", "p9","v8", "p8","v7", "p7","v6", "p6","v5", "p5","v4", "p4","v3", "p3","v2", "p2","v1", "p1"]]
y_test_inverse = []
for y in y_test:
    inverse = transformer.inverse_transform([[0, 0,0, 0,0, 0,0, 0,0, y[0],0, 0,0, 0,0, 0,0, 0,0,0]])
#     inverse = transformer.inverse_transform([[0, 0,0, 0,y[0], 0,0, 0,0, 0]])
    y_inverse = inverse.flatten()[9]
    y_test_inverse.append(y_inverse)
 
y_predict = model.predict(x_test)
y_predict_inverse = []
for y in y_predict:
    inverse = transformer.inverse_transform([[0, 0,0, 0,0, 0,0, 0,0, y[0],0, 0,0, 0,0, 0,0, 0,0,0]])
#     inverse = transformer.inverse_transform([[0, 0,0, 0,y[0], 0,0, 0,0, 0]])
    y_inverse = inverse.flatten()[9]
    print(y_inverse)
    y_predict_inverse.append(y_inverse)
 
 
plt.plot(y_test_inverse,'ro')
plt.plot(y_predict_inverse,'bo')
plt.xlabel('Time Period')
plt.ylabel('Close')
plt.show()