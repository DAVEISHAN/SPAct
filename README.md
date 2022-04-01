# SPAct
Official code repository for SPAct: Self-supervised Privacy Preservation for Action Recognition [CVPR-2022]

Work in progress ...

### Dataset preparation

UCF101: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
<br/>HMDB51: http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
<br/>VISPR: https://tribhuvanesh.github.io/vpa/
<br/>PA-HMDB51: https://github.com/TAMU-VITA/PA-HMDB51
<br/>LSHVU dataset: https://github.com/holistic-video-understanding/HVU-Dataset

### Intialization of networks
``cd initialization`` <br/>
To run initialization training for anonymization function: 
```
  python train_recon.py --run_id="give_any_expname_you_like"
  # add --restart argument to continue the stopped training
 ```


### Training of Anonymization function
TODO: Add code

### Evaluation of learned anonymization function
TODO: Add code

### Anonymization Visualization
TODO: Add code

### Citation

If you find the repo useful for your research, please consider citing our paper: 
```
@inproceedings{spact,
  title={SPAct: Self-supervised Privacy Preservation for Action Recognition},
  author={Dave, Ishan Rajendrakumar and Chen, Chen and Shah, Mubarak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
### Useful code repositories

[1] Privacy preserving action recognition (Wu et al., TPAMI 2020): https://github.com/VITA-Group/Privacy-AdversarialLearning 
<br/>[2] PA-HMDB annoatations https://github.com/VITA-Group/PA-HMDB51
<br/>[3] PyTorch Implementation of UNet: https://github.com/milesial/Pytorch-UNet
<br/>[4] Torchvision models: https://github.com/pytorch/vision/tree/main/torchvision/models
