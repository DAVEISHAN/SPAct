import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import config as cfg
import random
import pickle
import parameters_BL as params
import json, glob
# import math
# import cv2
# from tqdm import tqdm
import time
import torchvision
import torchvision.transforms as trans
# from decord import VideoReader

class vispr_ssl(Dataset):

    def __init__(self, datasplit, shuffle = True, data_percentage = 1.0):
        # self.labeled_datapaths = open(os.path.join(cfg.path_folder,'10percentTrain_crcv.txt'),'r').read().splitlines()
        self.datasplit = datasplit
        if self.datasplit == 'train':
            self.datapath = os.path.join(cfg.vispr_path, 'train2017')
            self.all_paths = glob.glob(self.datapath + '/*.jpg')
            self.labels = pickle.load(open(os.path.join(cfg.label_paths , 'train_labels.pkl'), 'rb'))
        elif self.datasplit == 'test':
            self.datapath = os.path.join(cfg.vispr_path, 'test2017')
            self.labels = pickle.load(open(os.path.join(cfg.label_paths , 'test_labels.pkl'), 'rb'))

            self.all_paths = glob.glob(self.datapath + '/*.jpg')                    
        
        self.classes= json.load(open(cfg.class_mapping))
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19

        # self.label_json = {}



    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        img1, img2, label, vid_path = self.process_data(index)
        return img1, img2, label, vid_path

    def process_data(self, idx):
    
        # label_building
        img_path = self.data[idx]
        # print(vid_path)
        # exit()
        label = self.labels[os.path.basename(img_path).replace('.jpg','')]
                
        #self.classes[vid_path.split('/')[6]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS  

        
        # clip_building
        img1, img2 = self.build_image(img_path)

        return img1, img2, label, img_path

    def build_image(self, img_path):

        try:
            img = torchvision.io.read_image(img_path)
            if img.shape[0] == 1:
                # print(img.shape)
                img = img.repeat(3, 1, 1)
            if not img.shape[0]==3:
                # print(f'{img_path} has {img.shape[0]} channels')
                return None, None
            # print(img.shape)
            # exit()
            ori_reso_w = img.shape[-1]
            ori_reso_h = img.shape[1]

            random_array = np.random.rand(2,8)
            x_erase = np.random.randint(0,params.reso_h, size = (2,))
            y_erase = np.random.randint(0,params.reso_w, size = (2,))


            cropping_factor1 = np.random.uniform(0.6, 1, size = (2,)) # on an average cropping factor is 80% i.e. covers 64% area
            x0 = np.random.randint(0, ori_reso_w - ori_reso_w*cropping_factor1[0] + 1) 
            y0 = np.random.randint(0, ori_reso_h - ori_reso_h*cropping_factor1[0] + 1)

            contrast_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            hue_factor1 = np.random.uniform(-0.05,0.05, size = (2,))
            saturation_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            brightness_factor1 = np.random.uniform(0.9,1.1,size = (2,))
            gamma1 = np.random.uniform(0.85,1.15, size = (2,))


            erase_size1 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
            erase_size2 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
            random_color_dropped = np.random.randint(0,3,(2))



            if self.datasplit == 'train':
                img1 = self.augmentation(img, random_array[0], x_erase[0], y_erase[0], cropping_factor1[0], x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],                           gamma1[0],erase_size1[0],erase_size2[0], random_color_dropped[0])/255.0
                img2 = self.augmentation(img, random_array[1], x_erase[1], y_erase[1], cropping_factor1[1], x0, y0, contrast_factor1[1], hue_factor1[1], saturation_factor1[1], brightness_factor1[1],                           gamma1[0],erase_size1[1],erase_size2[1], random_color_dropped[1])/255.0

            elif self.datasplit == 'test':
                img = self.test_augmentation(img)/255.0
            
            try:
                assert(len(img1.shape)!=0)
                assert(len(img2.shape)!=0)
                return img1, img2
            except:
                # print(frames_full)
                print(f'Image {img_path} Failed')
                return None, None   

        except:
            print(f'Image {img_path} Failed')
            return None, None

    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        # ori_reso_h = image.shape[]
        ori_reso_h,ori_reso_w = image.shape[1], image.shape[-1]

        image = trans.functional.resized_crop(image,y0,x0,int(ori_reso_h*cropping_factor1),int(ori_reso_w*cropping_factor1),(params.reso_h,params.reso_w))


        if random_array[0] < 0.125/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[1] < 0.3/2 :
            image = trans.functional.adjust_hue(image, hue_factor = hue_factor1) # hue factor will be between [-0.25, 0.25]*0.4 = [-0.1, 0.1]
        if random_array[2] < 0.3/2 :
            image = trans.functional.adjust_saturation(image, saturation_factor = saturation_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[3] < 0.3/2 :
            image = trans.functional.adjust_brightness(image, brightness_factor = brightness_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[4] > 0.9:
            image = trans.functional.rgb_to_grayscale(image, num_output_channels = 3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1) #gamma range [0.8, 1.2]
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        # image = trans.functional.to_tensor(image)

        if random_array[6] < 0.5/2 :
            image = trans.functional.erase(image, x_erase, y_erase, erase_size1, erase_size2, v=0) 

        return image


    def test_augmentation(self, image):
        h,w = image.shape[1], image.shape[-1]
        side = min(h,w)
        image = trans.functional.center_crop(image, side)
        image = trans.functional.resize(image,(params.reso_h,params.reso_w))
        return image



def collate_fn1(batch):
    clip, label, vid_path = [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            # clip.append(torch.from_numpy(np.asarray(item[0],dtype='f')))
            clip.append(torch.stack(item[0],dim=0)) 

            label.append(item[1])
            vid_path.append(item[2])

    clip = torch.stack(clip, dim=0)

    return clip, label, vid_path

def collate_fn2(batch):

    f_clip, label, vid_path, frame_list = [], [], [], []
    # print(len(batch))
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) # I might need to convert this tensor to torch.float
            label.append(item[1])
            vid_path.append(item[2])
            frame_list.append(item[3])

        # else:
            # print('oh no2')
    # print(len(f_clip))
    f_clip = torch.stack(f_clip, dim=0)
    
    return f_clip, label, vid_path, frame_list 
            
def collate_fn_train(batch):

    img1, img2, label, vid_path = [], [], [], []
    # print(len(batch))
    for item in batch:
        if not (item[0] == None or item[-1] == None):
            img1.append(item[0]) # I might need to convert this tensor to torch.float
            img2.append(item[1])
            label.append(item[2])
            vid_path.append(item[3])
        # else:
            # print('oh no2')
    # print(len(f_clip))
    img1 = torch.stack(img1, dim=0)
    img2 = torch.stack(img2, dim=0)

    return img1, img2, label, vid_path 

if __name__ == '__main__':

    train_dataset = vispr_ssl(datasplit = 'train', shuffle = False, data_percentage = 1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)

    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t=time.time()

    for i, (img1, img2, label, vid_path) in enumerate(train_dataloader):
        if i%10 == 0:
            print()
            # clip = clip.permute(0,1,3,4,2)
            print(f'Full_clip shape is {img1[0]}')
            print(f'Full_clip shape is {img2[0]}')

            print(f'Label is {label}')
            # pickle.dump(clip, open('f_clip.pkl','wb'))
            # pickle.dump(label, open('label.pkl','wb'))
            # exit()
    print(f'Time taken to load data is {time.time()-t}')

    # train_dataset = multi_baseline_dataloader_val_strong(shuffle = False, data_percentage = 1.0,  mode = 4)
    # train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)

    # print(f'Step involved: {len(train_dataset)/params.batch_size}')
    # t=time.time()

    # for i, (clip, label, vid_path, _) in enumerate(train_dataloader):
    #     if i%25 == 0:
    #         print()
    #         # clip = clip.permute(0,1,3,4,2)
    #         print(f'Full_clip shape is {clip.shape}')
    #         print(f'Label is {label}')
    #         # print(f'Frame list is {frame_list}')
            
    #         # pickle.dump(clip, open('f_clip.pkl','wb'))
    #         # pickle.dump(label, open('label.pkl','wb'))
    #         # exit()
    # print(f'Time taken to load data is {time.time()-t}')

