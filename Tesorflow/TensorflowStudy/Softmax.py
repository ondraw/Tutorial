import tensorflow as tf
import numpy as np
from numpy import shape
from pickletools import optimize
from pyasn1.compat import integer
from time import time
import random
from absl.logging import log
from pyasn1.compat.octets import null

#0 : 일반
#1 : sparse_softmax_cross_entropy_with_logits 사용.
#2 : ann 

CompileType = 2
  
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
  
  
# 모델을 적용할때 float32가 아니면 에러가 난다.
x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
x_train,x_test = x_train/50.,x_test/50.
  
# sparse_softmax_cross_entropy_with_logits을 사용하지 않으면 on_hot으로 처리해야 한다.
if CompileType == 0 or CompileType == 2 : #softmax_cross_entropy_with_logits 에서 one-hot이여야 한다.
    y_train, y_test = tf.one_hot(y_train, depth=4), tf.one_hot(y_test, depth=4)
  
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
if CompileType == 2 :
    train_data = train_data.repeat().shuffle(6000).batch(64)
else :
    train_data = train_data.repeat().shuffle(6000).batch(10)
train_data_iter = iter(train_data)
  
  
W = tf.Variable(tf.zeros(shape=[3,4]))
b = tf.Variable(tf.zeros(shape=[4]))
  
  
writer = tf.summary.create_file_writer('./Work.log/{}'.format(time()))
tf.summary.trace_on(graph=True)
  
@tf.function
def softmax_regression(x):
  logits = tf.matmul(x, W) + b
  return tf.nn.softmax(logits),logits
  
@tf.function
def cross_entropy_loss(logits, y):
    #sparse_softmax_cross_entropy_with_logits을 하지 않으면 이렇게 해준다.
    if CompileType == 0 :
        return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(logits), axis=[1]))
    elif CompileType == 2:
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) #y는 one-hot이어야 한다.
         
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
  
  
hW1 = tf.Variable(tf.random.normal(shape=[3,64]))
hB1 = tf.Variable(tf.random.normal(shape=[64]))
hW2 = tf.Variable(tf.random.normal(shape=[64,64]))
hB2 = tf.Variable(tf.random.normal(shape=[64]))
oW = tf.Variable(tf.random.normal(shape=[64,4]))
oB = tf.Variable(tf.random.normal(shape=[4]))
     
@tf.function
def ann(x):
    hOut1 = tf.nn.relu(tf.matmul(x,hW1) + hB1)
    hOut2 = tf.nn.relu(tf.matmul(hOut1,hW2) + hB2)
    #return tf.nn.softmax((tf.matmul(hOut2,oW) + oB)
    return tf.matmul(hOut2,oW) + oB

if CompileType == 2 :
    optimize = tf.optimizers.Adam(0.001)
else : 
    optimize = tf.optimizers.SGD(0.5)
 
  
@tf.function
def train_step(x,y):
    with tf.GradientTape() as tape:
        if CompileType == 0 or CompileType == 1:
            y_pred,logits = softmax_regression(x)
            #sparse_softmax_cross_entropy_with_logits을 하지 않으면 이렇게 해준다.
            if CompileType == 0 :
                loss = cross_entropy_loss(y_pred, y)
            else :
                loss = cross_entropy_loss(logits, y)
        else:
            logits = ann(x)
            loss = cross_entropy_loss(logits, y)
             
    with writer.as_default():
        tf.summary.scalar('loss',loss,step=optimize.iterations)
          
    if CompileType == 2:
        gradient = tape.gradient(loss,[hW1,hB1,hW2,hB2,oW,oB])
        optimize.apply_gradients(zip(gradient,[hW1,hB1,hW2,hB2,oW,oB]))
    else :
        gradient = tape.gradient(loss,[W,b])
        optimize.apply_gradients(zip(gradient,[W,b]))
      
      
for i in range(1000):
    batch_xs,batch_ys = next(train_data_iter)
    train_step(batch_xs,batch_ys)
      
#CompileType == 0일때 결과가 이상하게 나온다.
#주의 결과가 2행으로 나온다 왜그런가는 나중에 알아봐야 한다. 알아보고 코멘트 쳐라.
if CompileType == 0:
    result_y = softmax_regression(x_test)[0]
    correct_prediction = tf.equal(tf.argmax(result_y,1), tf.argmax(y_test,1))
elif CompileType == 2 :
    result_y = ann(x_test)
    correct_prediction = tf.equal(tf.argmax(result_y,1), tf.argmax(y_test,1))
else:
    result_y = softmax_regression(x_test)[0]
    correct_prediction = tf.equal(tf.argmax(result_y,1), y_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('확률 %f' % accuracy)
          