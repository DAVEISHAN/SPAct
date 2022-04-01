import torch
# from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time
import os
import numpy as np
# from model import build_r3d_classifier, load_r3d_classifier
import parameters_BL as params
import config as cfg
from dl_reconstruction import *
import sys, traceback
# from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
# import cv2
from torch.utils.data import DataLoader
import math
import argparse
import itertools
from unet_model import UNet
import warnings
from torchvision.utils import save_image

warnings.filterwarnings("ignore", category=DeprecationWarning) 

from keras.utils import to_categorical

# if torch.cuda.is_available(): 
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True

def train_epoch(run_id, epoch, data_loader, fa_model, criterion, optimizer, writer, use_cuda,learning_rate2):
    print('train at epoch {}'.format(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr']=learning_rate2
        writer.add_scalar('Learning Rate', learning_rate2, epoch)  

        print("Learning rate is: {}".format(param_group['lr']))
  
    losses, weighted_losses = [], []
    # loss_mini_batch = 0
    # optimizer.zero_grad()

    fa_model.train()

    for i, (inputs, vid_path) in enumerate(data_loader):
        # print(f'label is {label}')
        # inputs = inputs.permute(0,4,1,2,3)
        # print(inputs.shape, flush= True)
        optimizer.zero_grad()

        # inputs = inputs.permute(0,2,1,3,4)
        # print(inputs.shape)
        
        # inputs = inputs.permute(0,2,1,3,4)
        if use_cuda:
            inputs = inputs.cuda()
        
        # print(inputs.shape)
        output = fa_model(inputs)
        loss = criterion(output,inputs)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 500 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush = True)
        
    print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
    writer.add_scalar('Training Loss', np.mean(losses), epoch)

    del loss, inputs, output

    return fa_model, np.mean(losses)

# validation_epoch(run_id, epoch, train_dataloader, fa_model,  criterion, optimizer, writer, use_cuda, learning_rate2)
# def val_epoch(run_id, epoch,mode, skip, hflip, cropping_fac, pred_dict,label_dict, data_loader, fa_model,  criterion, writer, use_cuda):
    
def validation_epoch(run_id, epoch, data_loader, fa_model,  criterion, optimizer, writer, use_cuda, learning_rate2):
    fa_model.eval()

    losses = []
    # predictions, ground_truth = [], []
    vid_paths = []
    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    for i, (inputs, vid_path) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        # inputs = inputs.permute(0,4,1,2,3)
        if len(inputs.shape) != 1:

            # inputs = inputs.permute(0, 2, 1, 3, 4)

            if use_cuda:
                inputs = inputs.cuda()
            
            # print(inputs.shape)
            with torch.no_grad():
        
            
                output = fa_model(inputs)
                
                loss = criterion(output,inputs)
                losses.append(loss.item())

            # print(len(predictions))

        if i % 300 == 0: 
            print(f'Validation Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush = True)
            vis_image = torch.cat([inputs[:5], output[:5]], dim=0)

            save_image(vis_image, save_dir+'/combined'+'_epoch_'+ str(epoch) + '_batch_'+ str(i)  + '.png', padding=5, nrow=5)

    del inputs, output, loss 
    
    print('Validation Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
    writer.add_scalar('Validation Loss', np.mean(losses), epoch)
    return np.mean(losses)
    
def train_classifier(run_id, restart, saved_model):
    use_cuda = True
    best_score = 10000
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fa_model = UNet(n_channels = 3, n_classes=3)
    epoch0 = 0
   

    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d58/model_best_e154_loss_0.7567.pth'
    learning_rate1 = params.learning_rate
    
    # temp = list(np.linspace(0,1, 10) + 1e-9) + [1 for i in range(50)] + [0.1 for i in range(100)]

    # lr_array = l*np.asarray(temp)
    learning_rate2 = learning_rate1 

    
    # print(lr_array[:50])
    criterion= nn.L1Loss()

    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        fa_model=nn.DataParallel(fa_model)
        criterion.cuda()
        fa_model.cuda()
    else:
        print('Only 1 GPU is available')
        criterion.cuda()
        fa_model.cuda()


    optimizer = optim.Adam(fa_model.parameters(),lr=params.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.80)
    train_dataset = reconstruction_dataloader(datasplit = 'train', shuffle = False, data_percentage = 1.0)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
    # val_array = list(range(250))#[0,1,25,50,100,150,175] + list(range(200,255))
    # val_array = [0,9,60] + [90, 93, 96, 99] + [125+ x for x in range(100)]
    val_array = [0,3,5,10,12,15, 20, 25, 30, 35, 40, 45] + [50+ x for x in range(100)]


    modes = list(range(params.num_modes))
    skip = list(range(1,params.num_skips+1))
    hflip = params.hflip#list(range(2))
    cropping_fac1 = params.cropping_fac1#[0.7,0.85,0.8,0.75]
    print(f'Num modes {len(modes)}')
    print(f'Num skips {skip}')
    print(f'Cropping fac {cropping_fac1}')
    print(f'Base learning rate {params.learning_rate}')
    print(f'Scheduler patient {params.lr_patience}')
    print(f'Scheduler drop {params.scheduled_drop}')

    modes, skip,hflip, cropping_fac =  list(zip(*itertools.product(modes,skip,hflip,cropping_fac1)))
    accuracy = 0
    lr_flag1 = 0
    lr_counter = 0
    # train_loss_prev = 1000
    # lr_array = [0.001, 0.01, 0.1, 1] + [1 for x in range(15)] + [0.5 for x in range(15)] +  [0.1 for x in range(30)] + [0.05 for x in range(20)] + [0.01 for x in range(25)]
    # lr_array = np.asarray(lr_array)*learning_rate1
    learning_rate1 = params.learning_rate
    train_loss_best = 1000

    for epoch in range(epoch0, params.num_epochs):
        if epoch < params.warmup and lr_flag1 ==0:
            learning_rate2 = params.warmup_array[epoch]*params.learning_rate
        # learning_rate2 = lr_array[epoch]

        print(f'Epoch {epoch} started')
        start=time.time()
        try:
            fa_model, train_loss = train_epoch(run_id, epoch, train_dataloader, fa_model,  criterion, optimizer, writer, use_cuda, learning_rate2)
            
            # if train_loss > train_loss_prev:
            #     lr_counter += 1
            
            # if lr_counter > params.lr_patience:
            #     lr_counter = 0
            #     learning_rate2 = learning_rate2/params.scheduled_drop
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #     print(f'Learning rate dropping to its {scheduled_drop}th value to {learning_rate2} at epoch {epoch}')
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            # train_loss_prev = train_loss

            if train_loss > train_loss_best:
                lr_counter += 1
            if lr_counter > params.lr_patience:
                lr_counter = 0
                learning_rate2 = learning_rate2/params.scheduled_drop
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Learning rate dropping to its {params.scheduled_drop}th value to {learning_rate2} at epoch {epoch}')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')



            if train_loss_best < train_loss:
                train_loss_best = train_loss

            # if train_loss < 0.8 and lr_flag1 ==0:
            #     lr_flag1 =1 
            #     learning_rate2 = learning_rate1/2
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            # if train_loss < 0.4 and lr_flag1 ==1:
            #     lr_flag1 = 2  
            #     learning_rate2 = learning_rate1/10
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            # if train_loss < 0.25 and lr_flag1 ==2:
            #     lr_flag1 = 3
            #     learning_rate2 = learning_rate1/20
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            # if train_loss < 0.20 and lr_flag1 ==3:
            #     lr_flag1 = 4
            #     learning_rate2 = learning_rate1/100
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            # if train_loss < 0.05 and lr_flag1 ==4:
            #     lr_flag1 = 5
            #     learning_rate2 = learning_rate1/1000
            #     print(f'Dropping learning rate to {learning_rate2} for epoch')

            if epoch in val_array:
                validation_dataset = reconstruction_dataloader(datasplit = 'test', shuffle = False, data_percentage = 1.0)

                validation_dataloader = DataLoader(validation_dataset, batch_size=params.batch_size, \
                shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)

                val_loss = validation_epoch(run_id, epoch, train_dataloader, fa_model,  criterion, optimizer, writer, use_cuda, learning_rate2)

            if val_loss < best_score:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, 'model_{}_bestAcc_{}.pth'.format(epoch, str(val_loss)[:6]))
                states = {
                    'epoch': epoch + 1,
                    'lr_counter' : lr_counter,
                    # 'arch': params.arch,
                    'fa_model_state_dict': fa_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
                best_score = val_loss
            # else:
            save_dir = os.path.join(cfg.saved_models_dir, run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                'lr_counter' : lr_counter,
                # 'arch': params.arch,
                'fa_model_state_dict': fa_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            # scheduler.step()
            # elif epoch % 20 == 0:
            #     save_dir = os.path.join(cfg.saved_models_dir, run_id)
            #     save_file_path = os.path.join(save_dir, 'model_{}_Acc_{}_F1_{}.pth'.format(epoch, str(accuracy)[:6],str(f1_score)[:6]))
            #     states = {
            #         'epoch': epoch + 1,
            #         # 'arch': params.arch,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #     }
            #     torch.save(states, save_file_path)

            # scheduler.step()
        except:
            print("Epoch ", epoch, " failed")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        train_dataset = reconstruction_dataloader(datasplit = 'train', shuffle = False, data_percentage = 1.0)

        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
        print(f'Train dataset length: {len(train_dataset)}')
        print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_recon",
                        help='run_id')
    parser.add_argument("--restart", action='store_true')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None,
                        help='run_id')

    # print()
    # print('Repeating r3d57, Optimizer grad inside each iteration')
    # print()

    args = parser.parse_args()
    print(f'Restart {args.restart}')

    run_id = args.run_id
    saved_model = args.saved_model

    train_classifier(str(run_id), args.restart, saved_model)


        


