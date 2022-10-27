import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import sys
import shutil
import numpy as np
import cv2
from natsort import natsorted

def convert_onechannel(img):
  if len(img.shape)>2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img= cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
  return img

def check_channel(img):
  if img.shape[2]>3:
    img=img[:,:,:4]
  return img

def convert_tobgr(img):
  if len(img.shape)<2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  return img


class SegmentationBase(Dataset):
    def __init__(self,
                 file_path,img_dim=256,data_flip=True,with_abnormality=True
                 ):
        self.data_path = file_path
        self.segm_zippath=os.path.join(file_path, "Segmentation.zip")
        self.expert_zippath=os.path.join(file_path, "Expert.zip")
        self.radio_zippath=os.path.join(file_path, "Radiographs.zip")
        self.img_dim=img_dim
        if not self.img_dim:
           self.img_dim=256
        self.data_flip=data_flip
        self.with_abnormality=with_abnormality
        self.path_img=os.path.join(file_path, "Radiographs")
        self.path_label_teeth=os.path.join(file_path, "Segmentation/teeth_mask")
        self.path_label_mandibular=os.path.join(file_path, "Segmentation/maxillomandibular")
        self.path_label_abnormal=os.path.join(file_path, "Expert/mask")


        missing_data_error="Check your data path . It needs zips files or their folder of the data."  
        if not os.path.isdir(file_path):
            print("ERROR : Invalid Data Path : {} .. Exiting...".format(file_path))
            sys.exit()
        if not os.path.isfile(self.segm_zippath) :
            error="ERROR :Segmentation  zip file  {} is missing.. Exiting...".format(self.segm_zippath)
            if not os.path.isdir(self.path_label_teeth) or not os.path.isdir(self.path_label_mandibular):
              print(error)
              print("ERROR : Segmentation  folder   {} or {} is missing.. Exiting...".format(self.path_label_teeth,self.path_label_mandibular))
              print(missing_data_error)
              sys.exit()
        if not os.path.isfile(self.expert_zippath)  :
            error="ERROR :Expert  zip file   {} is missing.. Exiting...".format(self.expert_zippath)
            if not os.path.isdir(self.path_label_abnormal):
              print(error)
              print("ERROR :Expert  folder  {} is missing.. Exiting...".format(self.path_label_abnormal))
              print(missing_data_error)
              sys.exit()
        if not os.path.isfile(self.radio_zippath):
            error="ERROR :Radiographs zip file {} is missing. Exiting...".format(self.radio_zippath)
            if not os.path.isdir(self.path_img):
                print(error)
                print("ERROR :Radiographs folder {}  is missing. Exiting...".format(self.path_img))
                print(missing_data_error)
                sys.exit()   
        
  
        
        if not os.path.isdir(self.path_label_teeth) or not os.path.isdir(self.path_label_mandibular):
          shutil.unpack_archive(self.segm_zippath,file_path)

        if not os.path.isdir(self.path_label_abnormal):
          shutil.unpack_archive(self.expert_zippath,file_path)
        
        if not os.path.isdir(self.path_img):
          shutil.unpack_archive(self.radio_zippath,file_path)
        
        self.dirs_label_teeth=natsorted(os.listdir(self.path_label_teeth))
 
    def __len__(self):
        return len(self.dirs_label_teeth)

    def read_label(self,path,size):
        
        label = cv2.imread(path)
        label=convert_onechannel(label)
        label=cv2.resize(label, (size, size), interpolation= cv2.INTER_LINEAR )
        label=np.reshape(label,(size,size,1)) 
        return label
    
    def read_img(self,path,size):

      img = cv2.imread(path)
      img=convert_tobgr(img)
      img=check_channel(img)
      img=cv2.resize(img, (size, size), interpolation= cv2.INTER_LINEAR )
      img=np.reshape(img,(size,size,3))
      img=np.float32((img - 127.5) / 127.5 )
      return img
    
    def len(self):
        return len(self.dirs_label_teeth)
    
    def make_categoricalonehotlabelmap(self,mandibular,teeth,abnormal):

        categoricallabelmap=np.ones((self.img_dim,self.img_dim,1))

        mandibular=(mandibular/255)<1
        categoricallabelmap=np.where(mandibular,categoricallabelmap,2)

        teeth=(teeth/255)<1
        categoricallabelmap=np.where(teeth,categoricallabelmap,3)

        if self.with_abnormality:
          abnormal=(abnormal/255)<1
          categoricallabelmap=np.where(abnormal,categoricallabelmap,4)
        #else:
         # output = np.eye(3)[categoricallabelmap]
        output=categoricallabelmap[:,:,0]
        output=np.int32(output)
        output=np.eye(5)[output]

        return output

    def apply_flip(self,image,segmentationmap,label,categoricallabelmap):

        flipped_image=np.fliplr(image)
        flipped_segmentationmap=np.fliplr(segmentationmap)
        flipped_label=np.fliplr(label)
        flipped_categoricallabelmap=np.fliplr(categoricallabelmap)
        return flipped_image,flipped_segmentationmap,flipped_label,flipped_categoricallabelmap

  
    def make_segmentationmap(self,categorical_map):
        segmentationmap=np.zeros((self.img_dim,self.img_dim,3))
        segmentationmap[:,:,0]=categorical_map[:,:,3]
        if self.with_abnormality:
          segmentationmap[:,:,1]=categorical_map[:,:,4]          
        segmentationmap[:,:,2]=categorical_map[:,:,2]
        segmentationmap=np.float32((segmentationmap - 0.5) /0.5 )
        return segmentationmap
        
    def __getitem__(self, i):
        files_names=self.dirs_label_teeth
        path_img=os.path.join(self.path_img,files_names[i].upper())
        path_label_teeth=os.path.join(self.path_label_teeth,files_names[i])
        path_label_mandibular=os.path.join(self.path_label_mandibular,files_names[i])
        path_abnormal=os.path.join(self.path_label_abnormal,files_names[i].upper())
        image=self.read_img(path_img,self.img_dim)
        teeth=self.read_label(path_label_teeth, self.img_dim)
        mandibular=self.read_label(path_label_mandibular, self.img_dim)
        abnormal=self.read_label(path_abnormal, self.img_dim)
        categorical_map=self.make_categoricalonehotlabelmap(mandibular, teeth, abnormal)
        example = { "image": image,
                    "segmentation": categorical_map,
                     "relative_file_path_": files_names[i].upper(),
                    "file_path_": path_img,
                    "segmentation_path" : path_label_teeth
                          }
        return example


class TeethSegTrain(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(file_path="data/Tufts_Raw_Train",img_dim=size,data_flip=True,with_abnormality=True)
        
class TeethSegEval(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(file_path="data/Tufts_Raw_Val",img_dim=size,data_flip=True,with_abnormality=True)

class TeethSegTest(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(file_path="data/Tufts_Raw_Test",img_dim=size,data_flip=True,with_abnormality=True)