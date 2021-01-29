import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import utils
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np
import joblib

#normalize한 정보값을 읽어온다.
transformer = joblib.load('./Work.ethereum_price_regression_model_transformer.pkl') 
#모델을 불러온다.
model = load_model('./Work.ethereum_price_regression_model.h5')

##########모델 예측
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

inverse = transformer.inverse_transform([[0, 0, 0, y_predict.flatten()[0]]])
print(inverse.flatten()[-1])