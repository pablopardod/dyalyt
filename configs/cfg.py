from types import SimpleNamespace
import torch
import augmentations as A

import pandas as pd
import numpy as np

#Extraxt xyz_landmarks
body_parts = [2, 3, 4, 6, 7, 8, 10, 11, 12, 18, 19, 20, 26, 27, 28, 34, 35, 36, 42, 43, 44, 54, 55, 56, 58, 59, 60, 70, 71, 72, 74, 75, 76, 82, 83, 84, 90, 91, 92]
columns_upper_body = body_parts 
columns_upper_body.sort()
f = 'driveandact/openpose_3d/vp1/run1b_2018-05-29-14-02-47.ids_1.openpose.3d.csv'
columns = pd.read_csv(f).columns
xyz_landmarks = np.array(columns)[columns_upper_body]
print('🕺 Used landmarks:')
print(xyz_landmarks)

#class Config:
Config = SimpleNamespace(
    MAX_LEN       = 100,
    DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    NUM_WORKERS   = 2,
    PIN_MEMORY    = True  ,

    #ENCODER Config
    n_landmarks = 13,  
    ce_ignore_index = -100,
    label_smoothing = 0.,
    return_logits = False,
    pretrained = True,
    val_mode = 'padded',
    dim = 208,
    input_dim=208,
    encoder_dim=208,
    num_layers=14,
    num_attention_heads= 4,
    feed_forward_expansion_factor=1,
    conv_expansion_factor= 2,
    input_dropout_p= 0.1, 
    feed_forward_dropout_p= 0.1,  
    attention_dropout_p= 0.1, 
    conv_dropout_p= 0.1, 
    conv_kernel_size= 51,
    
    #Decoder
    num_classes = 12, 
    num_classes_fine = 34,  
    num_classes_coarse = 12,

    #Augmentations
    decoder_mask_aug = 0.2,
    flip_aug = 0.5,
    outer_cutmix_aug = 0.5,
    
    pose_augmentations = A.Compose([    A.HeightPerson(sample_rate=(0.5,1.5), p=0.8),
                                        A.WidthPerson(sample_rate=(0.7,1.2), p=0.6),
                                        A.RotatePerson(sample_rate=(-10.0,10.0), p=0.6)]),
    
    pose_drop = A.Compose([A.OneOf([
                                        A.PoseDrop2(landmarks = xyz_landmarks,mask_value=0.,p=0.7),
                                        A.FaceDrop(landmarks = xyz_landmarks,mask_value=0.,p=0.5),],p=0.5),]),

    train_custom = A.Compose([
                                A.Resample(sample_rate=(0.5,1.4), p=0.8), 
                                A.DynamicResample(sample_rate=(0.9,1.1),windows=(20,30), p=0.25),
                                A.TimeShift(p=0.5),
                                A.CutVideo(limits=(15,15)), 
                                A.OneOf([
                                    A.TemporalMask(size=(0.2,0.4),mask_value=0.,p=0.5),
                                    A.TemporalMask(size=(0.1,0.2),num_masks = (2,3),mask_value=0.,p=0.5),
                                    A.TemporalMask(size=(0.05,0.1),num_masks = (4,5),mask_value=0.,p=0.5)],p=0.5),
                                A.SpatialMask(size=(0.05,0.1),mask_value=0.,mode='relative',p=0.5), #mask with 0 as it is post-normalization
                                ]),
   
    val_aug = None,

    #Drive&Act Dataloader Configuration
    dataset='Drive&Act',
    data_path = './driveandact/',
    label_type_aad= 'coarse',
    convert_coord = 0,
    noise_augment = 0,
    noise_std = 0.001,
    loss = 'cross_entropy'

    )