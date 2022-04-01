import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import config as cfg
import random
import pickle
import parameters_BL as params
import json
import math
import cv2
from tqdm import tqdm
import time
import torchvision.transforms as trans
import traceback
# from decord import VideoReader

class baseline_dataloader_train_strong(Dataset):

    def __init__(self, shuffle = True, data_percentage = 1.0, downsample = 1.0):
        # self.labeled_datapaths = open(os.path.join(cfg.path_folder,'10percentTrain_crcv.txt'),'r').read().splitlines()
        self.labeled_datapaths = open(os.path.join(cfg.path_folder,'ucf101_labeled.txt'),'r').read().splitlines()
        self.unlabled_datapaths = open(os.path.join(cfg.path_folder,'ucf101_unlabeled.txt'),'r').read().splitlines()
        self.blur = params.blur 
        self.blackened = params.blackened
        self.all_paths = self.labeled_datapaths + self.unlabled_datapaths
        self.classes= json.load(open(cfg.class_mapping))['classes']
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.downsample = downsample
        self.erase_size = 19*self.downsample 
        


    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path = self.process_data(index)
        return clip, label, vid_path

    def process_data(self, idx):
    
        # label_building
        if params.blur:
            vid_path = self.data[idx].replace('/home/ishan/self_supervised/UCF101', cfg.blur_path_folder)
        elif params.blackened:
            vid_path = self.data[idx].replace('/home/ishan/self_supervised/UCF101', cfg.blackened_path_folder)
        else:
            vid_path = self.data[idx]

        # print(vid_path)
        # exit()
        label = self.classes[vid_path.split('/')[-2]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS  
        
        # clip_building
        clip = self.build_clip(vid_path)

        return clip, label, vid_path

    def build_clip(self, vid_path):

        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(7)

            ############################# frame_list maker start here#################################

            # start_frame = np.random.randint(0,int(params.num_frames)) # here 0 can be replaced with random starting point, but, for now it's getting complicated

            # skip_max = int((frame_count - start_frame)/params.num_frames)

            # sr_full = skip_max #np.random.randint(1,skip_max+1)

            # frames_full = [start_frame] + [start_frame + i*sr_full for i in range(1,params.num_frames)]
            skip_frames_full = params.fix_skip #frame_count/(params.num_frames)

            left_over = frame_count - params.fix_skip*params.num_frames

            # start_frame_full = np.random.randint(0,int(left_over)) # here 0 can be replaced with random starting point, but, for now it's getting complicated
            if left_over <=0:
                start_frame_full = 0
            else:    
                start_frame_full = np.random.randint(0,int(left_over))


            frames_full = start_frame_full + np.asarray([int(int(skip_frames_full)*f) for f in range(params.num_frames)])

            # print(frames_full)
            # exit()

            if frames_full[-1] >= frame_count:
                # print('some corner case not covered')
                frames_full[-1] = int(frame_count-1)
            ################################ frame list maker finishes here ###########################

            ################################ actual clip builder starts here ##########################
            full_clip = []
            list_full = []
            count = -1
            random_array = np.random.rand(2,8)
            x_erase = np.random.randint(0,params.reso_h, size = (2,))
            y_erase = np.random.randint(0,params.reso_w, size = (2,))


            cropping_factor1 = np.random.uniform(0.6, 1, size = (2,)) # on an average cropping factor is 80% i.e. covers 64% area
            x0 = np.random.randint(0, params.ori_reso_w*self.downsample  - params.ori_reso_w*self.downsample *cropping_factor1[0] + 1) 
            y0 = np.random.randint(0, params.ori_reso_h*self.downsample  - params.ori_reso_h*self.downsample *cropping_factor1[0] + 1)

            contrast_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            hue_factor1 = np.random.uniform(-0.05,0.05, size = (2,))
            saturation_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            brightness_factor1 = np.random.uniform(0.9,1.1,size = (2,))
            gamma1 = np.random.uniform(0.85,1.15, size = (2,))


            erase_size1 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
            erase_size2 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
            random_color_dropped = np.random.randint(0,3,(2))

            while(cap.isOpened()): 
                count += 1
                ret, frame = cap.read()
                if ((count not in frames_full)) and (ret == True): 
                    continue
                if ret == True:
                    if (count in frames_full):
                        full_clip.append(self.augmentation(frame, random_array[0], x_erase[0], y_erase[0], cropping_factor1[0],\
                                x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],\
                                gamma1[0],erase_size1[0],erase_size2[0], random_color_dropped[0]))
                        list_full.append(count)
                else:
                    break

            if len(full_clip) < params.num_frames and len(full_clip)>(params.num_frames/2) :
                # print(f'Clip {vid_path} is missing {params.num_frames-len(full_clip)} frames')
                remaining_num_frames = params.num_frames - len(full_clip)
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
            
            try:
                assert(len(full_clip)==params.num_frames)

                return full_clip
            except:
                # print(frames_full)
                print(f'Clip {vid_path} Failed-1')
                traceback.print_exc()
                return None   

        except:
            print(f'Clip {vid_path} Failed-2')
            traceback.print_exc()

            return None

    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        image = self.PIL(image)
        image = trans.functional.resize(image, (int(image.size[0]*self.downsample), int(image.size[1]*self.downsample)))

        image = trans.functional.resized_crop(image,y0,x0,int(params.ori_reso_h*self.downsample *cropping_factor1),int(params.ori_reso_h*self.downsample *cropping_factor1),(params.reso_h,params.reso_w))


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
            image = trans.functional.to_grayscale(image, num_output_channels = 3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1) #gamma range [0.8, 1.2]
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        image = trans.functional.to_tensor(image)

        if random_array[6] < 0.5/2 :
            image = trans.functional.erase(image, x_erase, y_erase, erase_size1, erase_size2, v=0) 

        return image


class multi_baseline_dataloader_val_strong(Dataset):

    def __init__(self, shuffle = True, data_percentage = 1.0, mode = 0, skip = 1, \
                hflip=0, cropping_factor=1.0, downsample = 1.0):
        self.labeled_datapaths = open(os.path.join(cfg.path_folder,'ucf101_test.txt'),'r').read().splitlines()
        self.all_paths = self.labeled_datapaths
        self.classes= json.load(open(cfg.class_mapping))['classes']
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.mode = mode
        self.skip = skip
        self.hflip = hflip
        self.cropping_factor = cropping_factor
        self.downsample = downsample                 
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list

    def process_data(self, idx):
    
        # label_building
        if params.blur:
            vid_path = self.data[idx].replace('/home/ishan/self_supervised/UCF101', cfg.blur_path_folder)
        elif params.blackened:
            vid_path = self.data[idx].replace('/home/ishan/self_supervised/UCF101', cfg.blackened_path_folder)
        else:
            vid_path = self.data[idx]        
        # print(vid_path)
        label = self.classes[vid_path.split('/')[-2]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS  
        
        # clip_building
        clip, frame_list = self.build_clip(vid_path)

        return clip, label, vid_path, frame_list

    def build_clip(self, vid_path):
        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(7)

            N = frame_count
            n = params.num_frames

            # skip_max = int(N/n)
            # if (skip_max < 2) and (self.skip>1):
            #     # print(frame_count)
            #     # print('Skip_max =1 and already covered')
            #     return None


            skip_frames_full = params.fix_skip #int(skip_max/params.num_skips*self.skip)

            # if skip_frames_full<2:
            #     if self.skip==1:
            #         skip_frames_full = 1
            #     else:
            #         # print('Skip_rate already covered')
            #         return None
            
            left_over = skip_frames_full*n
            F = N - left_over

            start_frame_full = 0 + int(np.linspace(0,F-10,params.num_modes)[self.mode])

            # start_frame_full = int(F/(params.num_modes-1))*self.mode + 1 if params.num_modes>1 else 0
            if start_frame_full< 0:
                start_frame_full = self.mode
                # print()
                # print(f'oh no, {start_frame_full}')
                # return None, None

            full_clip_frames = []

            
            full_clip_frames = start_frame_full + np.asarray(
                [int(int(skip_frames_full) * f) for f in range(params.num_frames)])

            

            count = -1
            # fail = 0
            full_clip = []
            list_full = []

            while (cap.isOpened()):
                count += 1
                ret, frame = cap.read()
                # print(frame.shape)
                if ((count not in full_clip_frames) and (ret == True)):
                    # ret, frame = cap.read()
                    continue
                # ret, frame = cap.read()
                if ret == True:
                    # Resize - THIS CAN ALSO BE IMPROVED
                    # frame1 = cv2.resize(frame, (171, 128))
                    # print(f'new size{frame1.shape}')
                    # print()
                    # frame1 = self.augmentation(frame1[0:0 + 112, 29:29 + 112])

                    if (count in full_clip_frames):
                        full_clip.append(self.augmentation(frame))
                        list_full.append(count)

                else:
                    break

            if len(full_clip) < params.num_frames and len(full_clip)>(params.num_frames/2):
                # print()
                # print(f'Full clip has {len(full_clip)} frames')
                # print(f'Full video has {frame_count} frames')
                # print(f'Full video suppose to have these frames: {full_clip_frames}')
                # print(f'Actual video has           these frames: {list_full}')
                # print(f'final count value is {count}')
                # if params.num_frames - len(full_clip) >= 1:
                #     print(f'Clip {vid_path} is missing {params.num_frames - len(full_clip)} frames')
                # for remaining in range(params.num_frames - len(full_clip)):
                #     full_clip.append(frame1)
                remaining_num_frames = params.num_frames - len(full_clip)
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
            # print(len(full_clip))
            assert (len(full_clip) == params.num_frames)

            return full_clip, list_full

        except:
            print(f'Clip {vid_path} Failed')
            return None, None

    def augmentation(self, image):
        image = self.PIL(image)
        image = trans.functional.resize(image, (int(image.size[0]*self.downsample), int(image.size[1]*self.downsample)))
        if self.cropping_factor <= 1:
            image = trans.functional.center_crop(image,(int(params.ori_reso_h*self.downsample*self.cropping_factor),int(params.ori_reso_h*self.downsample*self.cropping_factor)))
        image = trans.functional.resize(image, (params.reso_h, params.reso_w))
        if self.hflip !=0:
            image = trans.functional.hflip(image)

        return trans.functional.to_tensor(image)


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

    f_clip, label, vid_path = [], [], [], 
    # print(len(batch))
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) # I might need to convert this tensor to torch.float
            label.append(item[1])
            vid_path.append(item[2])
        # else:
            # print('oh no2')
    # print(len(f_clip))
    f_clip = torch.stack(f_clip, dim=0)
    
    return f_clip, label, vid_path 

if __name__ == '__main__':

    train_dataset = baseline_dataloader_train_strong(shuffle = False, data_percentage = 1.0, downsample = 1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)

    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t=time.time()

    for i, (clip, label, vid_path) in enumerate(train_dataloader):
        if i%10 == 0:
            print()
            clip = clip.permute(0,1,3,4,2)
            print(f'Full_clip shape is {clip.shape}')
            print(f'Label is {label}')
            # pickle.dump(clip, open('f_clip.pkl','wb'))
            # pickle.dump(label, open('label.pkl','wb'))
            # exit()
    print(f'Time taken to load data is {time.time()-t}')

    # train_dataset = multi_baseline_dataloader_val_strong(shuffle = False, data_percentage = 1.0,  mode = 5)
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

