import numpy as np

num_frames = 16
num_workers = 10
batch_size = 32#16#24
learning_rate = 1e-4#1e-4#1e-5
num_epochs = 300
data_percentage = 1.0#.0
v_batch_size = 48#80

fix_skip = 2
num_modes = 5
num_skips = 1
hflip = [0] #list(range(2))
cropping_fac1 = [0.8] #[0.7,0.85,0.8,0.75]

reso_h = 112
reso_w = 112

ori_reso_h = 240
ori_reso_w = 320

sr_ratio = 4


warmup_array = list(np.linspace(0.01,1, 5) + 1e-9)
warmup = len(warmup_array)

num_classes = 102#401
lr_patience = 0
scheduled_drop = 5

num_pa = 7 #Number of privacy attributes


ft_loss_weight = 1 #1 is equal weight with fb

learning_rate_fa = learning_rate
learning_rate_fb = 1*learning_rate
learning_rate_ft = learning_rate

data_percentage_ucf101 = 1.0
batch_size_ucf101 = 24

data_percentage_vispr = 0.9578
batch_size_vispr = 64



