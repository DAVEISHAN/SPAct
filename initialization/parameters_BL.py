import numpy as np

num_frames = 16
num_workers = 10
batch_size = 32#16#24
learning_rate = 1e-3#1e-4#1e-5
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