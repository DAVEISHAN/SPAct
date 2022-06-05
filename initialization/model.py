import numpy as np 
import torch.nn as nn
import torch
import torchvision
from torchsummary import summary
from r3d import r3d_18
from r3d_classifier import r3d_18_classifier
from r3d import mlp
from i3d import InceptionI3d
import parameters_BL as params


# from r2p1d import r2plus1d_18, embedder
# from classifier_r2p1d import r2plus1d_18_classifier
# # from try1_model import r2plus1d_18_changed
# # from dilated_r2plus1d import r2plus1d_18
from torchvision.models.utils import load_state_dict_from_url

def ssl_pretrained_load():
#     given = torch.load('/home/c3-0/ishan/privacy_preserving1/BYOL_load/model_final_checkpoint_phase799.torch')
    r50 = torchvision.models.resnet50(pretrained = False, progress = False)
#     state_dict = r50.state_dict()
#     #The following is the backbone model loading

#     junk_trunk = '_feature_blocks.'                             
#     for key, value in given['classy_state_dict']['base_model']['model']['trunk'].items():
#         layer_name = key.replace('_feature_blocks.', '')
#         state_dict[layer_name] = value 
#     r50.load_state_dict(state_dict, strict = True)
    r50.fc = nn.Identity()
#     print('Backbone loaded successfully!')

    ############################ MLP building

    class MLP(nn.Module):

        def __init__(self, final_embedding_size = 128, use_normalization = True):
            
            super(MLP, self).__init__()

            self.final_embedding_size = final_embedding_size
            self.use_normalization = use_normalization
            # self.fc1 = nn.Linear(512*3*3,512)
            self.fc1 = nn.Linear(2048,2048, bias = True)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(2048, self.final_embedding_size, bias = True)
        def forward(self, x):
            x = self.relu(self.fc1(x))
            #print(f'After fc1, shape of x is {x.shape}')
            x = nn.functional.normalize(self.fc2(x), p=2, dim=1)
            return x

    mlp = MLP()
    state_dict = mlp.state_dict()
    # print(state_dict.keys())
    # exit()
    # given MLP ['0.clf.0.weight', '0.clf.0.bias', '1.clf.0.weight', '1.clf.0.bias']
    # my MLP has (['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])

    for key, value in given['classy_state_dict']['base_model']['model']['heads'].items():
        # print(key)
        layer_name = key.replace('0.clf.0', 'fc1')
        layer_name = layer_name.replace('1.clf.0', 'fc2')
        # print(layer_name)
        state_dict[layer_name] = value 
    mlp.load_state_dict(state_dict, strict = True)
    print('MLP loaded successfully!')
    combined = nn.Sequential(r50, mlp)
    return combined



def build_r3d_classifier(num_classes = 102, kin_pretrained = False, self_pretrained = True, saved_model_file = None):
    model = r3d_18_classifier(pretrained = kin_pretrained, progress = False)
    
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    if self_pretrained == True:
        pretrained = torch.load(saved_model_file)
        pretrained_kvpair = pretrained['state_dict']
        # print(pretrained_kvpair)
        # exit()

        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            if 'module.1.' in layer_name:
                continue
            elif layer_name[:2] == '1.':
                continue
            if layer_name[:2] != '0.':
                layer_name = layer_name.replace('module.0.','')
            else:
                layer_name = layer_name[2:]

            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=True)
        print(f'model {saved_model_file} loaded successsfully!')
    # exit()
    model.fc = nn.Linear(512, num_classes)
    return model 

def load_r3d_classifier(num_classes = 102, saved_model_file = None):
    model = r3d_18_classifier(pretrained = False, progress = False)

    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    model.fc = nn.Linear(512, num_classes)

    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['state_dict']
    # print(pretrained_kvpair)
    # exit()

    model_kvpair = model.state_dict()
    for layer_name, weights in pretrained_kvpair.items():
        layer_name = layer_name.replace('module.0.','')
        model_kvpair[layer_name] = weights   
    model.load_state_dict(model_kvpair, strict=True)
    print(f'model {saved_model_file} loaded successsfully!')
    return model 


def build_r3d_backbone():
    model = r3d_18(pretrained = False, progress = False)
    
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    return model

def build_r3d_mlp():
    f = build_r3d_backbone()
    g = mlp()
    model = nn.Sequential(f,g)
    return model
    
def load_r3d_mlp(saved_model_file):
    model = build_r3d_mlp()
    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['state_dict']
    model_kvpair = model.state_dict()

    for layer_name, weights in pretrained_kvpair.items():
        # if 'module.1' in layer_name: # removing embedder part which is module.1 in the model+embedder
        #     continue
        layer_name = layer_name.replace('module.','')
        model_kvpair[layer_name] = weights  

    model.load_state_dict(model_kvpair, strict=True)
    print(f'{saved_model_file} loaded successfully')
    
    return model 

def build_i3dori_classifier(num_classes =401, dropout_keep_prob = 0.5):
    model = InceptionI3d(num_classes = num_classes, dropout_keep_prob = dropout_keep_prob)
    return model

#This file is just for model loading
def model_building(restart, save_dir, saved_model, architecture):
    if restart:
            saved_model_file = save_dir + '/model_temp.pth'
            
            if os.path.exists(saved_model_file):
                model = load_r3d_classifier(saved_model_file= saved_model_file)
                epoch0 = torch.load(saved_model_file)['epoch']
            else:
                print(f'No such model exists: {saved_model_file} :(')
                if not (saved_model == None or len(saved_model) == 0):
                    print(f'Trying to load {saved_model[30:]}')
                    model = build_r3d_classifier(saved_model_file = saved_model[30:], num_classes = params.num_classes)
                else:
                    print(f'It`s a baseline experiment!')
                    model = build_r3d_classifier(self_pretrained = False, saved_model_file = None, num_classes = params.num_classes) 
                epoch0 = 0

    else:

        if not (saved_model == None or len(saved_model) == 0):
            print(f'Trying to load {saved_model[30:]}')
            if architecture == 'r3d18':
                model = build_r3d_classifier(saved_model_file = saved_model[30:], num_classes = params.num_classes)
            elif architecture == 'i3dori':
                model = build_i3dori_classifier(num_classes = params.num_classes)
        else:
            print(f'It`s a baseline experiment!')
            if architecture == 'r3d18':
                model = build_r3d_classifier(self_pretrained = False, saved_model_file = None, num_classes = params.num_classes)
            elif architecture == 'i3dori':
                model = build_i3dori_classifier(num_classes = params.num_classes)
        epoch0 = 0
    return model, epoch0

if __name__ == '__main__':
    # model = arch_r50_classifier(num_classes = 101)
    # model.eval()
    # model.cuda()
    # summary(model, (16, 3, 112, 112))
    input = torch.rand(5, 3, 16, 112, 112).cuda()
    # model = build_r3d_classifier(num_classes = 102, saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_best_e89_loss_15.573.pth')
    model = build_r3d_classifier(num_classes = 102, saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_best_e102_loss_15.418.pth')

    # sftp://ishan@crcv.eecs.ucf.edu/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_best_e102_loss_15.418.pth
    # model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)

    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/4self_1e-3/model_e108_loss_0.00.pth'
    # model1 = load_r2plus1d_18_dilatedv2(saved_model_file= saved_model_file)
    # model = nn.Sequential(*list(model.children()))
    # print(model)
    # print()
    # model = r2plus1d_18_dilated()
    # print(model)

    # # print()
    model.eval()
    model.cuda()
    output = model(input)
    print(input.shape)
    print(output.shape)
    # model1.eval()
    # model1.cuda()
    # output = model((input,'s'))

    # print(output[0].shape)
    # summary(model1, (3, 16, 112, 112))
    # summary(model, (1,(3, 16, 112, 112)))

