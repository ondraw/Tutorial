from __future__ import print_function
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense,Conv2D,MaxPool2D,Dropout,Conv2DTranspose,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from absl import app
from absl import flags
import tensorflow as tf
import numpy as np
import datetime
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import fcn.TensorflowUtils as utils
import fcn.read_MITSceneParsingData as scene_parsing
import fcn.BatchDatsetReader as dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from fcn.BatchFaceImageDatset import BatchFaceImageDatset

# 학습에 필요한 설정값들을 tf.flag.FLAGS로 지정합니다.
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", "2", "batch size for training")
flags.DEFINE_string("logs_dir", "Work.fcn2.logs/", "path to logs directory")
flags.DEFINE_string("data_dir", "Data_zoo/My", "path to dataset")
flags.DEFINE_float("learning_rate", "5e-5", "Learning rate for Adam Optimizer")
flags.DEFINE_string('mode', "train", "Mode train/ visualize")
# flags.DEFINE_string('mode', "visualize", "Mode train/ visualize")


# 학습에 필요한 설정값들을 지정합니다.
MAX_ITERATION = int(10120 + 1)
NUM_OF_CLASSESS = 151       # 레이블 개수
IMAGE_SIZE = 224


# Define custom loss
def custom_loss(layer):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        print('y_true ',y_true.shape)
        print('y_pred ',y_pred.shape)
        return 0.01
   
    # Return a function
    return loss

def main(_):
    
    pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    pre_trained_vgg.trainable = False
    
  
    tensorBoard = TensorBoard(log_dir='./Work.fcn2.log/{}'.format(time()))
    FCN_model = Sequential()
    FCN_model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    FCN_model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    FCN_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    FCN_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    FCN_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    FCN_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    FCN_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    FCN_model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


    FCN_model.add(Conv2D(filters=4096, kernel_size=7,strides=1, padding='same',activation='relu'))
    FCN_model.add(Dropout(0.2))
    FCN_model.add(Conv2D(filters=4096, kernel_size=7,strides=1, padding='same',activation='relu'))
    FCN_model.add(Dropout(0.2))
    FCN_model.add(Conv2D(filters=NUM_OF_CLASSESS, kernel_size=1  ,strides=1, padding='same',activation='relu'))
    FCN_model.add(Conv2DTranspose(filters=512, kernel_size=4,strides=2, padding='same',activation=None))
    FCN_model.add(Conv2DTranspose(filters=256, kernel_size=4,strides=2, padding='same',activation=None))
    FCN_model.add(Conv2DTranspose(filters=NUM_OF_CLASSESS, kernel_size=16,strides=8, padding='same',activation=None))
    FCN_model.compile(optimizer=Adam(lr=FLAGS.learning_rate), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    FCN_model.summary()


    #이미지 레코더를 만듭니다.
    DataSet = BatchFaceImageDatset(FLAGS.data_dir,IMAGE_SIZE,IMAGE_SIZE)
    train_images, train_annotations,valid_images, valid_annotations = DataSet.GetData()
    train_images = train_images.astype('float32') / 255.0
    valid_images = valid_images.astype('float32') / 255.0
    
    FCN_model.fit(train_images, 
                  train_annotations, 
                  batch_size=2,
                  epochs=1,
                  verbose=1,
                  validation_data=(valid_images, valid_annotations), 
                  callbacks=[tensorBoard,ModelCheckpoint(filepath='./Work.fcn2.h5',save_best_only=True)])
    
# main 함수를 실행합니다.
if __name__ == "__main__":
    app.run(main)
