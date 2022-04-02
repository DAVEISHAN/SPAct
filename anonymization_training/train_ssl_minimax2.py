#srun -c12 --mem-per-cpu=7G --pty -p gpu --gres=gpu:ampere:1 --pty bash
#sbatch -c12 --mem-per-cpu=7G -p gpu --wrap="python train_ssl_minimax2.py --run_id="" --kin_pretrained" --gres=gpu:ampere:1 --job-name="ours" --output=".out"


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time, random, os, cv2, math, argparse, itertools, warnings, sys, traceback
import numpy as np 


import parameters_SSL as params
import config as cfg
from keras.utils import to_categorical


from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dl_vispr_ssl import vispr_ssl, collate_fn_train
from dl_vispr import vispr_dataset_generator
from dl_vispr import collate_fn_train as collate_fn_train_regular

from dl_ft_nov import baseline_dataloader_train_strong, multi_baseline_dataloader_val_strong, collate_fn_train_ucf101, collate_fn_val_ucf101
from model import build_r3d_classifier, load_r3d_classifier, ssl_pretrained_load, load_unet, build_privacy_classifier, load_privacy_classifier, ssl_privacy_load
from nt_xent_original import NTXentLoss
from torchvision.utils import save_image


warnings.filterwarnings("ignore", category=DeprecationWarning) 

torch.backends.cudnn.benchmark = True


def train_epoch(run_id, epoch, data_loader_vispr, dataloader_ucf101, ft_model, fa_model, fb_model,  criterion_ft, criterion_fb, optimizer_fa, optimizer_fb, optimizer_ft, writer, use_cuda,learning_rate_fa, learning_rate_fb, learning_rate_ft):
    for param_group in optimizer_fa.param_groups:
        param_group['lr']=learning_rate_fa
    for param_group_fb in optimizer_fb.param_groups:
        param_group_fb['lr']=learning_rate_fb
    for param_group_ft in optimizer_ft.param_groups:
        param_group_ft['lr']=learning_rate_ft
    writer.add_scalar('Learning Rate', learning_rate_fa, epoch)  
    writer.add_scalar('Learning Rate', learning_rate_fb, epoch)  
    writer.add_scalar('Learning Rate', learning_rate_ft, epoch)  

    print("Learning rate of fa is: {}".format(param_group['lr']))
    print("Learning rate of fb is: {}".format(param_group_fb['lr']))
    print("Learning rate of ft is: {}".format(param_group_ft['lr']))

    losses_fa, losses_fb, losses_ft = [], [], []

    step = 1
    for i, (data1, data2) in enumerate(zip(data_loader_vispr, dataloader_ucf101)):
        if use_cuda:
            inputs_vispr = [data1[ii].cuda() for ii in range(2)]
            labels_vispr = torch.from_numpy(np.asarray(data1[2])).float().cuda()
            inputs_ucf101 = data2[0].cuda()
            labels_ucf101 = torch.from_numpy(np.asarray(data2[1])).cuda()
        # print(labels_vispr)
        # print(labels_ucf101)
        #step-1 update fa
        optimizer_fa.zero_grad()
        optimizer_fb.zero_grad()
        optimizer_ft.zero_grad()

        if step ==1:
            fa_model.train()
            ft_model.eval()
            fb_model.eval()
            
            output1 = [fb_model(fa_model(inputs_vispr[ii])) for ii in range(2)]

            ori_bs, ori_t, ori_c, ori_h, ori_w = inputs_ucf101.shape
            inputs_ucf101 = inputs_ucf101.view(-1, inputs_ucf101.shape[2], inputs_ucf101.shape[3], inputs_ucf101.shape[4])

            output2 = ft_model(fa_model(inputs_ucf101).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w).permute(0,2,1,3,4))


            con_loss_criterion = NTXentLoss(device = 'cuda', batch_size= output1[0].shape[0], temperature= 0.1, use_cosine_similarity= False)

            loss_fb = con_loss_criterion(output1[0], output1[1])
            loss_ft = criterion_ft(output2,labels_ucf101) 

            loss_fa = -loss_fb + params.ft_loss_weight*loss_ft
            losses_fa.append(loss_fa.item())
            # print('loss computed finished')

            loss_fa.backward()
            # print('backward finished')

            optimizer_fa.step()
            step = 2
            # print('step-1 finished')
            if i % 100 == 0: 
                print(f'Training Epoch {epoch}, Batch {i}, loss_fa: {np.mean(losses_fa) :.5f}', flush = True)

            continue

        if step == 2:
            fa_model.eval()
            ft_model.train()
            fb_model.train()

            output1 = [fb_model(fa_model(inputs_vispr[ii])) for ii in range(2)]

            ori_bs, ori_t, ori_c, ori_h, ori_w = inputs_ucf101.shape
            inputs_ucf101 = inputs_ucf101.view(-1, inputs_ucf101.shape[2], inputs_ucf101.shape[3], inputs_ucf101.shape[4])

            output2 = ft_model(fa_model(inputs_ucf101).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w).permute(0,2,1,3,4))

            con_loss_criterion = NTXentLoss(device = 'cuda', batch_size= output1[0].shape[0], temperature= 0.1, use_cosine_similarity= False)

            loss_fb = con_loss_criterion(output1[0], output1[1])

            loss_ft = criterion_ft(output2,labels_ucf101) 

            losses_ft.append(loss_ft.item())
            losses_fb.append(loss_fb.item())

            loss_fb.backward()
            optimizer_fb.step()
            
            loss_ft.backward()
            optimizer_ft.step()
            step = 1
            if (i-1) % 100 == 0: 
                print(f'Training Epoch {epoch}, Batch {i}, loss_fb: {np.mean(losses_fb) :.5f}, loss_ft: {np.mean(losses_ft) :.5f}', flush = True)
            continue
            
    print('Training Epoch: %d, loss_fa: %.4f, loss_fb: %.4f, loss_ft: %.4f' % (epoch,  np.mean(losses_fa), np.mean(losses_fb), np.mean(losses_ft)))
    writer.add_scalar('Training loss_fa', np.mean(losses_fa), epoch)
    writer.add_scalar('Training loss_fb', np.mean(losses_fb), epoch)
    writer.add_scalar('Training loss_ft', np.mean(losses_ft), epoch)

    del loss_fb, loss_ft, loss_fa,  inputs_vispr, inputs_ucf101,  output1, output2

    return fa_model, fb_model, ft_model

def val_visualization_fa_vispr(save_dir, epoch, validation_dataloader, fa_model):
    for i, (inputs, label, vid_path) in enumerate(validation_dataloader):
    # inputs = inputs.permute(0,4,1,2,3)
        if len(inputs.shape) == 1:
            continue

            # inputs = inputs.permute(0, 2, 1, 3, 4)

        
        inputs = inputs.cuda()
        # label1 = torch.from_numpy(np.asarray(label))[0].item()
        # label = classes_dict_rev[label1]
        # print(label1, label)
        # label = label.replace(' ', '_')
        # ori_bs, ori_t, ori_c, ori_h, ori_w = inputs.shape
        # inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
        
        # print(inputs.shape)
        # exit()
        # save_image(torch.flip(inputs[::2,:],dims = [1]) , 'input_' + label + '.png', padding=1, nrow=inputs.shape[0])
        image_full_name = save_dir + 'combined_e'+ str(epoch) + '.png'
        # print(inputs.shape)
        with torch.no_grad():

        
            outputs = fa_model(inputs)
            vis_image = torch.cat([inputs, outputs], dim=0)
            save_image(vis_image, image_full_name, padding=5, nrow=int(inputs.shape[0]))

            # if i ==0:
            break


def val_epoch_vispr(run_id, epoch, validation_dataloader, fa_model, fb_model, criterion, writer, use_cuda):
    
    fa_model.eval()
    fb_model.eval()
    losses = []
    predictions, ground_truth = [], []
    vid_paths = []
    label_dict, pred_dict = {}, {}

    for i, (inputs, label, vid_path) in enumerate(validation_dataloader):
        # inputs = inputs.permute(0,4,1,2,3)
        if len(inputs.shape) != 1:

            # inputs = inputs.permute(0, 2, 1, 3, 4)

            if use_cuda:
                inputs = inputs.cuda()
                label = torch.from_numpy(np.asarray(label)).float().cuda()
        
            # print(inputs.shape)
        
            
            # print(f'fa_model output shape is {output1.shape}')
            # output1 = output1.reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)
            # print(f'fa_model reshaped output shape is {output1.shape}')
            # exit()
            
            
            # random.sample(range(output1.shape[1]), 2)
            # print(output1[:,0,].shape, flush = True)

            # print(inputs.shape)
            with torch.no_grad():
        
            
                # output1 = fa_model(inputs)
                output1 = fa_model(inputs)
                output = fb_model(output1)

                # print(f'fa_model output shape is {output1.shape}')
                



                '''# print(label)
                if use_cuda:
                    inputs = inputs.cuda()
                    label = torch.from_numpy(np.asarray(label)).cuda()
                # print(label)

            
                with torch.no_grad():

                    output = model(inputs)
                    loss = criterion(output,label)'''
                loss = criterion(output,label)
                losses.append(loss.item())


            predictions.extend(output.cpu().data.numpy())
            vid_paths.extend(vid_path)
            ground_truth.extend(label.cpu().data.numpy())

            # print(len(predictions))


            if i+1 % 99 == 0:
                print("Validation Epoch ", epoch, " Batch ", i, "- Loss : ", np.mean(losses))
        
    del inputs, output, label, loss 
    print("Validation Epoch ", epoch, "- Loss : ", np.mean(losses))

    ground_truth = np.asarray(ground_truth)
    # predictions = np.asarray(predictions)

    prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, (np.array(predictions) > 0.5).astype(int))
    predictions = np.asarray(predictions)
    # try:
    #     print(f'GT shape before putting in ap: {ground_truth.shape}')
    #     print(f'pred shape before putting in ap: {predictions.shape}')
    # except:
    #     print(f'GT len before putting in ap: {len(ground_truth)}')
    #     print(f'pred len before putting in ap: {len(predictions)}')

    ap = average_precision_score(ground_truth, predictions, average=None)
    
    print(f'Macro f1 is {np.mean(f1)}')
    print(f'Macro prec is {np.mean(prec)}')
    print(f'Macro recall is {np.mean(recall)}')
    print(f'Classwise AP is {ap}')
    print(f'Macro AP is {np.mean(ap)}')


    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    return pred_dict, label_dict, np.mean(ap)

def val_epoch_ucf101(run_id, epoch,mode, skip, hflip, cropping_fac, pred_dict,label_dict, data_loader, fa_model, model, criterion, writer, use_cuda):
    print(f'validation at epoch {epoch} - mode {mode} - skip {skip} - hflip {hflip} - cropping_fac {cropping_fac}')
    
    model.eval()
    fa_model.eval()  
    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for i, (inputs, label, vid_path, _) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        ground_truth.extend(label)
        # inputs = inputs.permute(0,4,1,2,3)
        if len(inputs.shape) != 1:

            # inputs = inputs.permute(0, 2, 1, 3, 4)

            if use_cuda:
                inputs = inputs.cuda()
                label = torch.from_numpy(np.asarray(label)).cuda()
            ori_bs, ori_t, ori_c, ori_h, ori_w = inputs.shape
            inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
            # print(inputs.shape)
            with torch.no_grad():
        
            
                output1 = fa_model(inputs)
                # print(f'fa_model output shape is {output1.shape}')
                output1 = output1.reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)
                # print(f'fa_model reshaped output shape is {output1.shape}')
                # exit()
                output1 = output1.permute(0,2,1,3,4)
                output = model(output1)



                '''# print(label)
                if use_cuda:
                    inputs = inputs.cuda()
                    label = torch.from_numpy(np.asarray(label)).cuda()
                # print(label)

            
                with torch.no_grad():

                    output = model(inputs)
                    loss = criterion(output,label)'''
                loss = criterion(output,label)
                losses.append(loss.item())


            predictions.extend(nn.functional.softmax(output, dim = 1).cpu().data.numpy())
            # print(len(predictions))


            if i+1 % 99 == 0:
                print("Validation Epoch ", epoch , "mode", mode, "skip", skip, "hflip", hflip, " Batch ", i, "- Loss : ", np.mean(losses))
        
    del inputs, output, label, loss 

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) # Prediction with the most confidence is the first element here
    c_pred = pred_array[:,0] #np.argmax(predictions,axis=1).reshape(len(predictions))

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    print_pred_array = []

    # for entry in range(pred_array.shape[0]):
    #     temp = ''
    #     for i in range(5):
    #         temp += str(int(pred_array[entry][i]))+' '
    #     print_pred_array.append(temp)
    # print(f'check {print_pred_array[0]}')
    # results = open('Submission1.txt','w')
    # for entry in range(len(vid_paths)):
    #     content = str(vid_paths[entry].split('/')[-1] + ' ' + print_pred_array[entry])[:-1]+'\n'
    #     results.write(content)
    # results.close()
    
    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)
    
    # print(f'Correct Count is {correct_count}')
    print(f'Epoch {epoch}, mode {mode}, skip {skip}, hflip {hflip}, cropping_fac {cropping_fac}, Accuracy: {accuracy*100 :.3f}')
    return pred_dict, label_dict, accuracy, np.mean(losses)



def train_classifier(run_id, restart, saved_model= None, kin_pretrained= False):
    use_cuda = True
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fa_model = load_unet(saved_model_file = '/home/c3-0/ishan/privacy_preserving1/unet_reconstruction_training/saved_models/recon1/model_20_bestAcc_0.0100.pth') #UNet(n_channels = 3, n_classes=3)
    if kin_pretrained:
        ft_model = build_r3d_classifier(kin_pretrained= True, self_pretrained = False)
    else:
        ft_model =  load_r3d_classifier(saved_model_file= '/home/c3-0/ishan/privacy_preserving1/ucf101_with_unetrecon/saved_models_and_logs/model_50_bestAcc_0.6262.pth')

    fb_model = ssl_privacy_load(saved_model_file = '/home/c3-0/ishan/privacy_preserving1/vispr_with_unetreocon/saved_models/SSL_unet_recon/model_160.pth')

    # fb_model = load_privacy_classifier(saved_model_file = '/home/c3-0/ishan/privacy_preserving1/vispr_with_unetreocon/saved_models/vispr_unetrecon1/model_57_bestAP_0.6271.pth', arch = 'mv1')
    

    learning_rate_fa = params.learning_rate_fa
    learning_rate_fb = params.learning_rate_fb
    learning_rate_ft = params.learning_rate_ft

    # if kin_pretrained:
    #     learning_rate_fa = params.learning_rate_fa/10
    #     learning_rate_fb = params.learning_rate_fb/10
    #     learning_rate_ft = params.learning_rate_ft/10

    criterion_ft = nn.CrossEntropyLoss()
    criterion_fb = nn.BCEWithLogitsLoss()

    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        ft_model=nn.DataParallel(ft_model)
        fa_model=nn.DataParallel(fa_model)
        fb_model=nn.DataParallel(fb_model)
        ft_model.cuda()
        fa_model.cuda()
        fb_model.cuda()

    else:
        print('Only 1 GPU is available')
        ft_model.cuda()
        fa_model.cuda()
        fb_model.cuda()

    optimizer_fa = optim.Adam(fa_model.parameters(),lr=params.learning_rate_fa)
    optimizer_fb = optim.Adam(fb_model.parameters(),lr=params.learning_rate_fb)
    optimizer_ft = optim.Adam(ft_model.parameters(),lr=params.learning_rate_ft)
    

    train_dataset_ucf101 = baseline_dataloader_train_strong(shuffle = False, data_percentage = params.data_percentage_ucf101)
    # train_dataset = baseline_dataloader(shuffle = False, data_percentage = 0.1)

    train_dataloader_ucf101 = DataLoader(train_dataset_ucf101, batch_size=params.batch_size_ucf101, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train_ucf101)
    print(f'UCF101 Train dataset length: {len(train_dataset_ucf101)}')
    print(f'UCF101 Train dataset steps per epoch: {len(train_dataset_ucf101)/params.batch_size_ucf101}')


    train_dataset_vispr = vispr_ssl(datasplit = 'train', shuffle = False, data_percentage = params.data_percentage_vispr)
    # train_dataset = baseline_dataloader(shuffle = False, data_percentage = 0.1)

    train_dataloader_vispr = DataLoader(train_dataset_vispr, batch_size=params.batch_size_vispr, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    print(f'VISPR Train dataset length: {len(train_dataset_vispr)}')
    print(f'VISPR Train dataset steps per epoch: {len(train_dataset_vispr)/params.batch_size_vispr}')
    epoch0  = 0
    val_array = [0, 3, 5,10,15,20, 25, 30, 35, 40, 45] + [50+ x for x in range(100)]


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


    for epoch in range(epoch0, params.num_epochs):
        # if epoch < params.warmup and lr_flag1 ==0:
        #     learning_rate2 = params.warmup_array[epoch]*params.learning_rate

        print(f'Epoch {epoch} started')
        start=time.time()

        fa_model, fb_model, ft_model = train_epoch(run_id, epoch, train_dataloader_vispr, train_dataloader_ucf101, ft_model, fa_model, fb_model,  criterion_ft, criterion_fb, optimizer_fa, optimizer_fb, optimizer_ft, writer, use_cuda,learning_rate_fa, learning_rate_fb, learning_rate_ft)

        if epoch in val_array:
            pred_dict = {}
            label_dict = {}
            val_losses =[]

            validation_dataset = vispr_dataset_generator(datasplit = 'test', shuffle = False, data_percentage = params.data_percentage)

            validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train_regular)
            '''
            pred_dict, label_dict, macro_ap = val_epoch_vispr(run_id, epoch, validation_dataloader, fa_model, fb_model, criterion_fb, writer, use_cuda)
            '''
            val_visualization_fa_vispr(save_dir, epoch, validation_dataloader, fa_model)       
            
            pred_dict = {}
            label_dict = {}
            val_losses =[]

            for val_iter in range(len(modes)):
                try:
                    validation_dataset = multi_baseline_dataloader_val_strong(shuffle = True, data_percentage = params.data_percentage,\
                        mode = modes[val_iter], skip = skip[val_iter], hflip= hflip[val_iter], \
                        cropping_factor= cropping_fac[val_iter])
                    validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_val_ucf101)
                    if val_iter ==0:
                        print(f'Validation dataset length: {len(validation_dataset)}')
                        print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}') 
                        
                   

                    pred_dict, label_dict, accuracy, loss = val_epoch_ucf101(run_id, epoch,modes[val_iter],skip[val_iter],hflip[val_iter],cropping_fac[val_iter], \
                        pred_dict, label_dict, validation_dataloader, fa_model, ft_model, criterion_ft, writer, use_cuda)
                    val_losses.append(loss)

                    predictions1 = np.zeros((len(list(pred_dict.keys())),params.num_classes))
                    ground_truth1 = []
                    entry = 0
                    for key in pred_dict.keys():
                        predictions1[entry] = np.mean(pred_dict[key], axis =0)
                        entry+=1

                    for key in label_dict.keys():
                        ground_truth1.append(label_dict[key])
                    
                    pred_array1 = np.flip(np.argsort(predictions1,axis=1),axis=1) # Prediction with the most confidence is the first element here
                    c_pred1 = pred_array1[:,0]

                    correct_count1 = np.sum(c_pred1==ground_truth1)
                    accuracy11 = float(correct_count1)/len(c_pred1)

                    
                    print(f'Running Avg Accuracy is for epoch {epoch}, mode {modes[val_iter]}, skip {skip[val_iter]}, hflip {hflip[val_iter]}, cropping_fac {cropping_fac[val_iter]} is {accuracy11*100 :.3f}% ')  
                except:
                    print(f'Failed epoch {epoch}, mode {modes[val_iter]}, skip {skip[val_iter]}, hflip {hflip[val_iter]}, cropping_fac {cropping_fac[val_iter]} is {accuracy11*100 :.3f}% ')  

            val_loss = np.mean(val_losses)
            predictions = np.zeros((len(list(pred_dict.keys())),params.num_classes))
            ground_truth = []
            entry = 0
            for key in pred_dict.keys():
                predictions[entry] = np.mean(pred_dict[key], axis =0)
                entry+=1

            for key in label_dict.keys():
                ground_truth.append(label_dict[key])
            
            pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) # Prediction with the most confidence is the first element here
            c_pred = pred_array[:,0]

            correct_count = np.sum(c_pred==ground_truth)
            accuracy1 = float(correct_count)/len(c_pred)
            print(f'Correct Count is {correct_count} out of {len(c_pred)}')
            writer.add_scalar('Validation Loss', np.mean(val_loss), epoch)
            writer.add_scalar('Validation Accuracy', np.mean(accuracy1), epoch)
            
            print(f'Overall Ft Accuracy is for epoch {epoch} is {accuracy1*100 :.3f}% ')
            # file_name = f'RunID_{run_id}_Acc_{accuracy1*100 :.3f}_cf_{len(cropping_fac1)}_m_{params.num_modes}_s_{params.num_skips}.pkl'     
            # pickle.dump(pred_dict, open(file_name,'wb'))
            accuracy = accuracy1

        if epoch % 3 == 0:
            save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                # 'arch': params.arch,
                'fa_model_state_dict': fa_model.state_dict(),
                'fb_model_state_dict': fb_model.state_dict(),
                'ft_model_state_dict': ft_model.state_dict(),
            }
            torch.save(states, save_file_path)
        # We will save optimizer weights for each temp model, not all saved models to reduce the storage 
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        states = {
                'epoch': epoch + 1,
                # 'arch': params.arch,
                'fa_model_state_dict': fa_model.state_dict(),
                'fb_model_state_dict': fb_model.state_dict(),
                'ft_model_state_dict': ft_model.state_dict(),
                'optimizer_fa': optimizer_fa.state_dict(),
                'optimizer_fb': optimizer_fb.state_dict(),
                'optimizer_ft': optimizer_ft.state_dict(),
            }
        torch.save(states, save_file_path)
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        train_dataset_ucf101 = baseline_dataloader_train_strong(shuffle = False, data_percentage = params.data_percentage_ucf101)
        # train_dataset = baseline_dataloader(shuffle = False, data_percentage = 0.1)

        train_dataloader_ucf101 = DataLoader(train_dataset_ucf101, batch_size=params.batch_size_ucf101, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train_ucf101)
        print(f'UCF101 Train dataset length: {len(train_dataset_ucf101)}')
        print(f'UCF101 Train dataset steps per epoch: {len(train_dataset_ucf101)/params.batch_size_ucf101}')


        train_dataset_vispr = vispr_ssl(datasplit = 'train', shuffle = False, data_percentage = params.data_percentage_vispr)
        # train_dataset = baseline_dataloader(shuffle = False, data_percentage = 0.1)

        train_dataloader_vispr = DataLoader(train_dataset_vispr, batch_size=params.batch_size_vispr, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
        print(f'VISPR Train dataset length: {len(train_dataset_vispr)}')
        print(f'VISPR Train dataset steps per epoch: {len(train_dataset_vispr)/params.batch_size_vispr}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_ssl_minmax",
                        help='run_id')
    parser.add_argument("--restart", action='store_true')
    parser.add_argument("--kin_pretrained", action='store_true')


    args = parser.parse_args()
    print(f'Restart {args.restart}')

    print(f'kin_pretrained {args.kin_pretrained}')
    run_id = args.run_id

    train_classifier(str(run_id), args.restart, None, args.kin_pretrained)
