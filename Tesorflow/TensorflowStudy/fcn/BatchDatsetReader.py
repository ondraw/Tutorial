# -*- coding: utf-8 -*-

import numpy as np
#import scipy.misc as msc
import cv2
import tensorflow as tf

# 데이터를 배치 단위로 묶는 BatchDatset 클래스를 정의합니다.
class BatchDatset:
  files = []
  images = []
  annotations = []
  image_options = {}
  batch_offset = 0
  epochs_completed = 0

  def __init__(self, records_list, image_options={}):
    """
    arguments:
      records_list: 읽어들일 file records의 list
      (record 예제: {'image': f, 'annotation': annotation_file, 'filename': filename})
      image_options: 출력 이미지를 조정할 수 있는 옵션(dictionary 형태)
        가능한 옵션들:
        resize = True/ False (resize를 적용할지 말지를 결정)
        resize_size = size of output image (resize된 출력이미지의 크기 - bilinear resize를 적용합니다.)
    """
    print("Initializing Batch Dataset Reader...")
    print(image_options)
    #records_list = records_list[:4] #$$너무 많아서 나중에 잘되면 제거하자.
    self.files = records_list
    self.image_options = image_options
    self._read_images()

  # raw 인풋 이미지와 annoation된 타겟 이미지를 읽습니다.
  def _read_images(self):
    self.__channels = True
    self.images = np.array([self._transform(filename['image']) for filename in self.files])
    self.__channels = False
    #By Song axis = 3 -> 2로 변경하였다. 
    self.annotations = np.array([np.expand_dims(self._transform(filename['annotation']), axis=2) for filename in self.files])
    
   
    print (self.images.shape)
    print (self.annotations.shape)

  # 이미지에 변형을 가합니다.
  def _transform(self, filename):
      
    #By Song 함수가 Defrecated 되었다.
    #image = msc.imread(filename)
    #By Song 채널이 false일때는 Gray로 변경하였다. sparse_cross_entropy_loss차수가 부족하다. annotaions파일은 그레이로 변경하여도 무관하지 않을까?ㅍ
    if self.__channels :
        image = cv2.imread(filename,cv2.IMREAD_COLOR)
    else :
        image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    #print('image ',image.shape) 
    if self.__channels and len(image.shape) < 3:
      image = np.array([image for i in range(3)])

    # resize 옵션이 있으면 이미지 resiging을 진행합니다.
    if self.image_options.get("resize", False) and self.image_options["resize"]:
      resize_size = int(self.image_options["resize_size"])
      #By Song Defrecated 되었다.
      resize_image = cv2.resize(image, dsize=(resize_size, resize_size), interpolation=cv2.INTER_AREA)
      #resize_image = imageio.imresize(image,
      #                             [resize_size, resize_size], interp='nearest')
    else:
      resize_image = image
      
    return np.array(resize_image)

  # 인풋 이미지와 타겟 이미지를 리턴합니다.
  def get_records(self):
    return self.images, self.annotations

  # batch_offset을 리셋합니다.
  def reset_batch_offset(self, offset=0):
    self.batch_offset = offset

  # batch_size만큼의 다음 배치를 가져옵니다.
  def next_batch(self, batch_size):
    start = self.batch_offset
    self.batch_offset += batch_size
    # 한 epoch의 배치가 끝난 경우 batch index를 처음으로 다시 설정합니다. 
    if self.batch_offset > self.images.shape[0]:
      # 한 epoch이 끝났습니다.
      self.epochs_completed += 1
      print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
      # 데이터를 섞습니다.(Shuffle)
      perm = np.arange(self.images.shape[0])
      np.random.shuffle(perm)
      self.images = self.images[perm]
      self.annotations = self.annotations[perm]
      # 다음 epoch을 시작합니다.
      start = 0
      self.batch_offset = batch_size

    end = self.batch_offset
    return self.images[start:end], self.annotations[start:end]

  # 전체 데이터 중에서 랜덤하게 batch_size만큼의 배치 데이터를 가져옵니다.
  def get_random_batch(self, batch_size):
    indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
    return self.images[indexes], self.annotations[indexes]