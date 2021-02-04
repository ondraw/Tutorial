import numpy as np
import cv2
import os
import random

class BatchFaceImageDatset:
    FileList = []
    #Images = []
    #Annotations = []
    
    TrainImages = []
    TrainAnnotaions = []
    Train_Batch_Offset = 0
    Train_Epochs_Completed = 0
    
    ValidImages = []
    ValidAnnotaions = []
    Valid_Batch_Offset = 0
    Valid_Epochs_Completed = 0
    
    
    def __init__(self,Path,width,height,train = True):
        self.FileList = self.GetFaceFileList(Path)
        AnnotationList = [self.RenameAnnotaion(filenamex) for filenamex in self.FileList]
        self.SyncImageFiles(self.FileList,AnnotationList)
            
        if train:
               
            
            Images = np.array([self.ReadImage(filename,width,height) for filename in self.FileList])
            print("Image Length=" , len(Images) , " Shape = " ,Images.shape)
             
            
            Annotations = np.array([self.ReadImage(filename,width,height,1) for filename in AnnotationList])
            Annotations = np.expand_dims(Annotations, axis=3)
            print("Annotations Length=" , len(Annotations) , " Shape = " ,Annotations.shape)
            
            
            ImageLen = len(Images)
            ValidePos = ImageLen - (int)(ImageLen * 0.1)
            
            self.TrainImages = Images[:ValidePos]
            self.TrainAnnotaions = Annotations[:ValidePos]  
            self.ValidImages = Images[ValidePos:]
            self.ValidAnnotaions = Annotations[ValidePos:]
            
            self.Suffle()
            
            print("TrainImages Length=" , len(self.TrainImages) , " Shape = " ,self.TrainImages.shape)
            print("TrainAnnotaions Length=" , len(self.TrainAnnotaions) , " Shape = " ,self.TrainAnnotaions.shape)
        else:
            pos = random.randrange(1,len(self.FileList))
            AnnotationList = AnnotationList[pos:pos+3]
            self.FileList = self.FileList[pos:pos+3]
            Images = np.array([self.ReadImage(filename,width,height) for filename in self.FileList])
            Annotations = np.array([self.ReadImage(filename,width,height,1) for filename in AnnotationList])
            Annotations = np.expand_dims(Annotations, axis=3)
            self.ValidImages = Images
            self.ValidAnnotaions = Annotations  
            print("ValidImages Length=" , len(self.ValidImages) , " Shape = " ,self.ValidImages.shape)
            print("ValidAnnotaions Length=" , len(self.ValidAnnotaions) , " Shape = " ,self.ValidAnnotaions.shape)
        
    def GetData(self):
         return self.TrainImages,self.TrainAnnotaions,self.ValidImages,self.ValidAnnotaions
        
     # batch_size만큼의 다음 배치를 가져옵니다.
    def Next(self, batch_size):
        start = self.Train_Batch_Offset
        self.Train_Batch_Offset += batch_size
        
        if self.Train_Batch_Offset > self.TrainImages.shape[0]:
            # 한 epoch이 끝났습니다.
            self.Train_Epochs_Completed += 1
            print("에포크 완료: " + str(self.Train_Epochs_Completed))
            self.Suffle()
            start = 0
            self.Train_Batch_Offset = batch_size
        
        end = self.Train_Batch_Offset
        
        return self.TrainImages[start:end],self.TrainAnnotaions[start:end]
    
    def NextValid(self, batch_size):
        start = self.Valid_Batch_Offset
        self.Valid_Batch_Offset += batch_size
        
        if self.Valid_Batch_Offset > self.ValidImages.shape[0]:
            # 한 epoch이 끝났습니다.
            self.Valid_Epochs_Completed += 1
            print("에포크 밸리드  완료: " + str(self.Valid_Epochs_Completed))
            self.Suffle()
            start = 0
            self.Valid_Batch_Offset = batch_size
        
        end = self.Valid_Batch_Offset
        
        return self.ValidImages[start:end],self.ValidAnnotaions[start:end]
   
    def Suffle(self):
        perm = np.arange(self.TrainImages.shape[0])
        np.random.shuffle(perm)
        self.TrainImages = self.TrainImages[perm]
        self.TrainAnnotaions = self.TrainAnnotaions[perm]
        perm = np.arange(self.ValidImages.shape[0])
        np.random.shuffle(perm)
        self.ValidImages = self.ValidImages[perm]
        self.ValidAnnotaions = self.ValidAnnotaions[perm]
        
    #파일리스트를 소팅하여 가져온다.
    def GetFaceFileList(self,Path):
        FileList = []
        file_list = os.listdir(Path)
        file_list.sort(reverse=False)
        for suppath in file_list : 
            if not "." in suppath:
                ListSubDir = os.listdir(Path +  "/" + suppath)
                ListSubDir.sort()
                for a in ListSubDir:
                    FileList.append(Path +  "/" + suppath + "/" + a)
        print('file length = ' , len(FileList))
        return FileList

    
    def SyncImageFiles(self,Images,Annotations) :
        length = len(Annotations) - 1
        for i in range(length,0,-1):
            if not os.path.isfile(Annotations[i]):
                del Images[i]
                del Annotations[i]

    
    
    # raw 인풋 이미지와 annoation된 타겟 이미지를 읽습니다.
    def ReadImage(self,filename,width,height,Cahnnel=3):
        try:
            if Cahnnel != 1:
                image = cv2.imread(filename,cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
                
                #레이블을 축소하자.
                rows,cols = image.shape
                for i in range(rows):
                    for j in range(cols):
                        if image[i,j] == 29:
                            image[i,j] = 1
                        elif image[i,j] == 76:
                            image[i,j ] = 2
                        elif image[i,j] == 150:
                            image[i,j ] = 3
                        else:
                            print('not label ' + str(image[i,j ]))
                            
            if width != 0:
                image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
                
            
                            
        except Exception as e:
            print('예외가 발생했습니다.', filename, " ",e)
        return np.array(image)

     # image filder 정보를 사용하여 annotation경로를 만든다.
    def RenameAnnotaion(self,filename):
        arrToken = filename.split('/')
        arrToken[-1] = arrToken[-1].split('.')[0] + '.ppm'
        del(arrToken[-2])
        fileSum = '/'.join(arrToken)
        return fileSum
        


# data = BatchFaceImageDatset('/Users/songs/Extension/Tutorial/Tesorflow/TensorflowStudy/fcn/Data_zoo/My',225,225)
# data.Next(2)
# data.Next(2)






