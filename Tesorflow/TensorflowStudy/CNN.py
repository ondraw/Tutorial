# -*- coding: utf-8 -*-

"""
CIFAR-10 Convolutional Neural Networks(CNN) 예제
"""

import tensorflow as tf
from tensorflow.python.distribute import step_fn
from time import time


(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train,x_test = x_train.reshape([-1,784]).astype('float32')/255.0,x_test.reshape([-1,784]).astype('float32')/255.0
(y_train,y_test) = tf.one_hot(y_train,10),tf.one_hot(y_test,10)

 
# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(50)
train_data_iter = iter(train_data)


class CNN(object):
    def __init__(self):
        #컨볼루션 5*5커널 ,1채널 input, 32채널 output , 0.01~0.05 데이터.
        self.W_conv1 = tf.Variable(tf.random.truncated_normal(shape=[5,5,1,32],stddev=5e-2))
        self.b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
        #14*14*32 결과 ---
         
         
        #컨볼루션 5*5커널 ,32채널 input, 64채널 output, 0.01~0.05 데이터.
        self.W_conv2 = tf.Variable(tf.random.truncated_normal(shape=[5,5,32,64],stddev=5e-2))
        self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
        #7*7*64 결과 ---
        
        #Fully Connected Layer 7*7*64 -> [7*7*64,1024] 형태변환한다.
        self.W_fc1 = tf.Variable(tf.random.truncated_normal(shape=[7*7*64,1024],stddev=5e-2))
        self.b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))
        
        #Output 1024 -> 10개로 결과
        self.W_output = tf.Variable(tf.random.truncated_normal(shape=[1024,10],stddev=5e-2))
        self.b_output = tf.Variable(tf.constant(0.1,shape=[10]))
        
        
        
    def __call__(self,x):
        x_image = tf.reshape(x,[-1,28,28,1]) #28*28*1 채널 데이터로 변경한다.
        #컨볼류션 28*28*1 -> 28*28*32 변경됨
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,self.W_conv1,strides=[1,1,1,1],padding='SAME') + self.b_conv1) 
        #풀링 스트라이드 [2,2]이기 때문에 28*28*32 -> 14*14*32으로 변경됨.
        h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
        #컨볼리션 14*14*32 -> 14*14*64 로 변경한다.
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,self.W_conv2,strides=[1,1,1,1],padding='SAME') + self.b_conv2) 
        #풀링 14*14*64 -> 7*7*64로 변경한다.
        h_pool2= tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
        #7*7*64 -> 3136 
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,self.W_fc1) + self.b_fc1)
        
        #1024 -> 10 결과.    
        logits = tf.matmul(h_fc1,self.W_output) + self.b_output
        y_pred = tf.nn.softmax(logits)
        return y_pred,logits
    
logdir = './Work.log/{}'.format(time())
writer = tf.summary.create_file_writer(logdir)
    
@tf.function    
def cross_entropy_loss(logits,y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))

optimizer = tf.optimizers.Adam(0.0004)

@tf.function
def train_step(model,x,y):    
    with tf.GradientTape() as tape:
        y_pred, logits = model(x)
        loss = cross_entropy_loss(logits, y)
        
    with writer.as_default():
        tf.summary.scalar('loss', loss, step=optimizer.iterations)
    gradients = tape.gradient(loss, vars(model).values())
    optimizer.apply_gradients(zip(gradients, vars(model).values()))

@tf.function
def compute_accuracy(y_pred,y):
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    return accuracy
    



CNN_model = CNN()

SAVER_DIR = "./Work.cnnmodel"
ckpt = tf.train.Checkpoint(**vars(CNN_model))
#max_to_keep 파일 유지를 5개까지.
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=SAVER_DIR, max_to_keep=5) 


latest_ckpt = tf.train.latest_checkpoint(SAVER_DIR)
if latest_ckpt:
  ckpt.restore(latest_ckpt)
  tf.summary.trace_on(graph=True, profiler=False)
  print("테스트 데이터 정확도 (Restored) : %f" % compute_accuracy(CNN_model(x_test)[0], y_test))
  with writer.as_default():
      tf.summary.trace_export(name="restore_cnnmodel",step=0)
  exit()


for step in range(1500):
    batch_x, batch_y = next(train_data_iter)
    if step % 100 == 0:
        train_accuracy = compute_accuracy(CNN_model(batch_x)[0], batch_y)
        print("반복 %d , 정확도 %f"%(step,train_accuracy))
    train_step(CNN_model, batch_x, batch_y)
    
    
ckpt_manager.save(checkpoint_number=step)
    


tf.summary.trace_on(graph=True, profiler=False)    
print("정확도 %f"% compute_accuracy(CNN_model(x_test)[0], y_test))
with writer.as_default():
  tf.summary.trace_export(name="cnnmodel",step=0)
    


    
    
        
        
        
        
        
        
        
        
        