# -*- coding: utf-8 -*-
# TensorFlow 2.0을 이용한 FCN(Fully Convolutional Networks) 구현

from __future__ import print_function
from absl import app
from absl import flags
import tensorflow as tf
import numpy as np
import datetime
import os
import cv2
from fcn.BatchFaceImageDatset import BatchFaceImageDatset

# 학습에 필요한 설정값들을 tf.flag.FLAGS로 지정합니다.
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", "2", "batch size for training")
flags.DEFINE_string("logs_dir", "Work.fnc3/", "path to logs directory")
flags.DEFINE_string("data_dir", "Data_zoo/My", "path to dataset")
flags.DEFINE_float("learning_rate", "5e-5", "Learning rate for Adam Optimizer")
flags.DEFINE_string('mode', "visualize", "Mode train/ visualize")
# flags.DEFINE_string('mode', "visualize", "Mode train/ visualize")


# 학습에 필요한 설정값들을 지정합니다.
MAX_ITERATION = int(4500 + 1)
#NUM_OF_CLASSESS = 151       # 레이블 개수
NUM_OF_CLASSESS = 4       # 0~3레이블 개수
IMAGE_SIZE = 224

# VGGNet 그래프 구조를 구축합니다.
def VGGNet():
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.summary()
    outputs = []
    outputs.append(vgg.get_layer('block5_conv3').output)
    outputs.append(vgg.get_layer('block4_pool').output)
    outputs.append(vgg.get_layer('block3_pool').output)
    model = tf.keras.Model([vgg.input], outputs)
    model.summary()
    return model

VGGNet_model = VGGNet()

# tf.keras.Model을 이용해서 FCN 모델을 정의합니다.
class FCN(tf.keras.Model):
    def __init__(self, rate):
        super(FCN, self).__init__()
        self.pool_layer_5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv_layer_6 = tf.keras.layers.Conv2D(filters=4096, kernel_size=7,
                                                   strides=1, padding='same',
                                                   activation='relu')
        self.dropout_layer_6 = tf.keras.layers.Dropout(rate)
        self.conv_layer_7 = tf.keras.layers.Conv2D(filters=4096, kernel_size=1,
                                                   strides=1, padding='same',
                                                   activation='relu')
        self.dropout_layer_7 = tf.keras.layers.Dropout(rate)
        self.conv_layer_8 = tf.keras.layers.Conv2D(filters=NUM_OF_CLASSESS, kernel_size=1,
                                                   strides=1, padding='same',
                                                   activation='relu')
        
        self.deconv_layer_1 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4,
                                                              strides=2, padding='same',
                                                              activation=None)
        
        self.deconv_layer_2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4,
                                                              strides=2, padding='same',
                                                              activation=None)
        
        self.deconv_layer_3 = tf.keras.layers.Conv2DTranspose(filters=NUM_OF_CLASSESS, kernel_size=16,
                                                              strides=8, padding='same',
                                                              activation=None)
  
    def call(self, x, is_training=True):
        # 이미지에 Mean Normalization을 수행합니다.
        processed_image = tf.keras.applications.vgg16.preprocess_input(x)
        
        # VGGNet의 conv5(conv5_3), pool4, pool3 레이어를 불러옵니다.
        features = VGGNet_model(processed_image)
        
        # pool5를 정의합니다.
        pool5 = self.pool_layer_5(features[0])
        assert pool5.shape == (FLAGS.batch_size, 7, 7, 512)
        
        # conv6을 정의합니다.
        relu6 = self.conv_layer_6(pool5)
        relu_dropout6 = self.dropout_layer_6(relu6, training=is_training)
        assert relu_dropout6.shape == (FLAGS.batch_size, 7, 7, 4096)
        
        # conv7을 정의합니다. (1x1 conv)
        relu7 = self.conv_layer_7(relu_dropout6)
        relu_dropout7 = self.dropout_layer_7(relu7, training=is_training)
        assert relu_dropout7.shape == (FLAGS.batch_size, 7, 7, 4096)
        
        # conv8을 정의합니다. (1x1 conv)
        conv8 = self.conv_layer_8(relu_dropout7)
        assert conv8.shape == (FLAGS.batch_size, 7, 7, NUM_OF_CLASSESS)
        
        # FCN-8s를 위한 Skip Layers Fusion을 설정합니다.
        # conv8의 이미지를 2배 확대합니다.
        conv_t1 = self.deconv_layer_1(conv8)
        # 2x conv8과 pool4를 더해 fuse_1 이미지를 만듭니다.
        fuse_1 = tf.add(conv_t1, features[1], name="fuse_1")
        assert fuse_1.shape == (FLAGS.batch_size, 14, 14, 512)
        
        # fuse_1 이미지를 2배 확대합니다.
        conv_t2 = self.deconv_layer_2(fuse_1)
        
        # 2x fuse_1과 pool3를 더해 fuse_2 이미지를 만듭니다.
        fuse_2 = tf.add(conv_t2, features[2], name="fuse_2")
        assert fuse_2.shape == (FLAGS.batch_size, 28, 28, 256)
        
        # fuse_2 이미지를 8배 확대합니다.
        conv_t3 = self.deconv_layer_3(fuse_2)
        assert conv_t3.shape == (FLAGS.batch_size, 224, 224, NUM_OF_CLASSESS)
        
        # 최종 prediction 결과를 결정하기 위해 마지막 activation들 중에서 argmax로 최대값을 가진 activation을 추출합니다.
        annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")
        assert annotation_pred.shape == (FLAGS.batch_size, 224, 224)
        
        return tf.expand_dims(annotation_pred, axis=3), conv_t3

# sparse cross-entropy 손실 함수를 정의합니다.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def sparse_cross_entropy_loss(logits, annotation):
#     print('annotaion ',annotation.shape)  #2,224,224,1
#     print('annotation len',len(annotation))  #2,224,224,1
#     print('logits ',logits.shape)         #2,224,224,151  
    return tf.reduce_mean(loss_object(tf.squeeze(annotation, axis=[3]), logits))

# 최적화를 위한 function을 정의합니다.
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        y_pred, logits = model(x)
        loss = sparse_cross_entropy_loss(logits, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
  
def save_image(image, save_dir, name):
    rows,cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if image[i,j] == 1:
                image[i,j] = 29
            elif image[i,j] == 2:
                image[i,j ] = 76
            elif image[i,j] == 3:
                image[i,j ] = 150
            else:
                print('not label ' + str(image[i,j ]))
    cv2.imwrite(os.path.join(save_dir, name + ".png"), image)

def main(_):
    # FCN 그래프를 선언합니다.
    FCN_model = FCN(rate=0.15)
    
    # 최적화를 위한 Adam 옵티마이저를 정의합니다.
    optimizer = tf.optimizers.Adam(FLAGS.learning_rate)
    
  
    print("tf.train.Checkpoint")
    ckpt = tf.train.Checkpoint(model=FCN_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=FLAGS.logs_dir, max_to_keep=5)
    summary_writer = tf.summary.create_file_writer(FLAGS.logs_dir)

    # 저장된 ckpt 파일이 있으면 저장된 파라미터를 불러옵니다.
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.logs_dir)
    if latest_ckpt:
        ckpt.restore(latest_ckpt)
        print("Model restored...")

  

    if FLAGS.mode == "train":
        DataSet = BatchFaceImageDatset(FLAGS.data_dir,IMAGE_SIZE,IMAGE_SIZE)
    
        for itr in range(MAX_ITERATION):
            # 학습 데이터를 불러옵니다.
            train_images, train_annotations = DataSet.Next(FLAGS.batch_size)
            # 이미지를 float32 타입으로 변환합니다.
            train_images = train_images.astype('float32')
            
            # train_step을 실행해서 파라미터를 한 스텝 업데이트합니다.
            train_step(FCN_model, train_images, train_annotations, optimizer)
            
            # 10회 반복마다 training 데이터 손실 함수를 출력합니다.
            if itr % 10 == 0:
                pred_annotation, logits = FCN_model(train_images)
                train_loss = sparse_cross_entropy_loss(logits, train_annotations)
                print("반복(Step): %d, Training 손실함수(Train_loss):%g" % (itr, train_loss))
                with summary_writer.as_default():
                    tf.summary.scalar('train_loss', train_loss, step=itr)
                    tf.summary.image("input_image", tf.cast(train_images, tf.uint8), step=itr, max_outputs=2)
                    tf.summary.image("ground_truth", tf.cast(train_annotations, tf.uint8), step=itr, max_outputs=2)
                    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), step=itr, max_outputs=2)
            
            # 500회 반복마다 validation 데이터 손실 함수를 출력하고 학습된 모델의 파라미터를 ckpt 파일로 저장합니다.
            if itr % 500 == 0:
                valid_images, valid_annotations = DataSet.NextValid(FLAGS.batch_size)
                # 이미지를 float32 타입으로 변환합니다.
                valid_images = valid_images.astype('float32')
                _, logits = FCN_model(valid_images, is_training=False)
                valid_loss = sparse_cross_entropy_loss(logits, valid_annotations)
                print("%s ---> Validation 손실함수(Validation_loss): %g" % (datetime.datetime.now(), valid_loss))
                ckpt_manager.save(checkpoint_number=itr)
                with summary_writer.as_default():
                    tf.summary.scalar('valid_loss', valid_loss, step=itr)

    elif FLAGS.mode == "visualize":
        DataSet = BatchFaceImageDatset(FLAGS.data_dir,IMAGE_SIZE,IMAGE_SIZE,False)
        # validation data로 prediction을 진행합니다.
        _,_,valid_images, valid_annotations = DataSet.GetData()
        
        # 이미지를 float32 타입으로 변환합니다.
        valid_images = valid_images.astype('float32')
        pred, _ = FCN_model(valid_images, is_training=False)
        FCN_model.summary()
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)
    
        # Input Data, Ground Truth, Prediction Result를 저장합니다.
        for itr in range(FLAGS.batch_size):
          save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
          save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
          save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
          print("Saved image: %d" % itr)
# main 함수를 실행합니다.
if __name__ == "__main__":
  app.run(main)
