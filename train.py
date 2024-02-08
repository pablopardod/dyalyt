import time
import random

import wandb 

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


import warnings
warnings.filterwarnings("ignore")

import augmentations as A

#Argparse
import argparse

def set_seed(seed):
    pl.seed_everything(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model,train_loader,val_loader,test_loader):

    # create run_id
    run_id = str(int(time.time()))
    checkpoint_path = './checkpoints/' + run_id

    #WanDB Logger
    if args.wandb == 'False':
        wandb_logger = False
    else:
        wandb.login()
        project_name = 'DYALYT'
        if args.wandb:
            run_name = args.wandb
            wandb_logger = WandbLogger(project=project_name,name=run_name)
        else:
            wandb_logger = WandbLogger(project=project_name)

    call = [ModelCheckpoint(
                dirpath=checkpoint_path + "/",
                save_weights_only=True,
                filename="{epoch:02d}-{val_acc_macro:.2f}",
                save_last=False,
                save_top_k=2,
                monitor="val_acc_macro",
                mode="max",
                save_on_train_epoch_end=False,
                every_n_epochs=1,
            ),
            EarlyStopping(monitor='val_acc_macro', patience=35, verbose=False, mode='max')]
            #LearningRateMonitor("epoch"),
        

    trainer = pl.Trainer(#default_root_dir=os.path.join(Config.CHECKPOINT_PATH),
                         default_root_dir=checkpoint_path,
                         accelerator="gpu" if str(Config.DEVICE).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=Config.epochs,
                         callbacks=call,
                         gradient_clip_val = Config.clip_grad, 
                         logger=wandb_logger,
                         deterministic=True)

    trainer.fit(model, train_loader, val_loader)
    
    #If Ctrl + C take the best model an test it
    if trainer.interrupted:
        print("Interrupted")
        best_run = trainer.checkpoint_callback.best_model_path
        print(f"Loading run: {best_run}")
        model = model.load_from_checkpoint(best_run)
        trainer.test(model, test_loader)
        model.finish_run()
        return model

    # Take the best run and test it
    best_run = trainer.checkpoint_callback.best_model_path
    #Take best run based on val_acc_macro of the name of the file
    print(f"Loading run: {best_run}")
    model = model.load_from_checkpoint(best_run)
    trainer.test(model, test_loader)

    model.finish_run()
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", type=str, default='False', help="log with wandb. 'False' to disable")
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.00487841507928422, help='learning rate')
parser.add_argument('--warmup', type=int, default=58, help='warmup percentage for CosineWarmupScheduler')
parser.add_argument('--max_len', type=int, default=100, help='max length of the sequence')
parser.add_argument('--split', type=int, default=0, choices = [0,1,2])
parser.add_argument('--config', type=str, default='cfg2aad', help='select config (model) for training (cfg1 or cfg2)')
parser.add_argument('--balanced',default=True, action='store_true', help='Balanced batch sampler')


#Dropout
parser.add_argument('--input_dropout_p', type=float, default=0.2484335126263782, help='input_dropout_p')
parser.add_argument('--feed_forward_dropout_p', type=float, default=0.16543905095482592, help='feed_forward_dropout_p')
parser.add_argument('--attention_dropout_p', type=float, default=0.25975470832480363, help='attention_dropout_p')
parser.add_argument('--conv_dropout_p', type=float, default=0.1855344530469696, help='conv_dropout_p')

#Optimizer
parser.add_argument('--weight_decay', type=float, default=0.2367514589905012, help='weight_decay')
parser.add_argument('--clip_grad', type=float, default=3.0, help='clip_grad')

#Model parameters
parser.add_argument('--input_dim', type=int, default=120, help='input_dim') #Not used
parser.add_argument('--encoder_dim', type=int, default=120, help='encoder_dim')
parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
parser.add_argument('--num_attention_heads', type=int, default=4, help='num_attention_heads')

#Data augmentation
parser.add_argument('--pose_aug', type=int, default=1, choices = [0,1])
parser.add_argument('--pose_drop', type=int, default=0, choices = [0,1])

args = parser.parse_args()

#Load dataset
from data.driveandact import CustomDataset
# Load trainer
from models.trainer import DDDNet
# Load config
from configs.cfg import Config 


#Apply args
Config.MAX_LEN = args.max_len
Config.lr = args.lr
Config.batch_size = args.batch_size 
Config.epochs = args.num_epochs
Config.warmup = args.warmup
Config.split = args.split

Config.input_dropout_p = args.input_dropout_p
Config.feed_forward_dropout_p = args.feed_forward_dropout_p
Config.attention_dropout_p = args.attention_dropout_p
Config.conv_dropout_p = args.conv_dropout_p

Config.weight_decay = args.weight_decay
Config.clip_grad = args.clip_grad

Config.input_dim = args.encoder_dim
Config.encoder_dim = args.encoder_dim 
Config.num_layers = args.num_layers
Config.num_attention_heads = args.num_attention_heads

#Augmentations
if args.pose_aug:
    aug = Config.pose_augmentations
else:
    aug = A.Compose([])

if args.pose_drop:
    drop = Config.pose_drop
else:
    drop = A.Compose([])

Config.train_custom = A.Compose([drop, aug, Config.train_custom])

print(f"Using {'GPU' if 'cuda' in str(Config.DEVICE) else 'CPU'}")

print('ℹ️  Label type: ',Config.label_type_aad)
print('ℹ️  Dataset: ',Config.dataset)

# To be reproducible
set_seed(12921)

# Set up the dataset and dataloader 
path = 'skeletons.pkl'

train_dataset = CustomDataset(Config.data_path, mode='train', split=Config.split, label_type=Config.label_type_aad, 
                                seq_len=Config.MAX_LEN, num_classes=Config.num_classes, convert_coord=Config.convert_coord, 
                                noise=Config.noise_augment, noise_std=Config.noise_std,aug=Config.train_custom, n_landmarks=Config.n_landmarks, path=path)
val_dataset = CustomDataset(Config.data_path, mode='val', split=Config.split, label_type=Config.label_type_aad, 
                                seq_len=Config.MAX_LEN, num_classes=Config.num_classes, convert_coord=Config.convert_coord, n_landmarks=Config.n_landmarks, path=path)
test_dataset = CustomDataset(Config.data_path, mode='test', split=Config.split, label_type=Config.label_type_aad, 
                                seq_len=Config.MAX_LEN, num_classes=Config.num_classes, convert_coord=Config.convert_coord, n_landmarks=Config.n_landmarks, path=path)


# Balanced batch sampler
if args.balanced:
    loss_weights = train_dataset.getLossWeights()
    weight_sample = torch.tensor([loss_weights[i] for i in train_dataset.labels]) #labels_fine -> labels
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_sample, len(train_dataset))
else:
    sampler = None

train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=Config.batch_size,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
        sampler=sampler,
    )

val_dataloader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=None,
        drop_last=True,
    )

test_dataloader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=None,
    )

Config.total_training_steps = Config.epochs * len(train_dataloader)
Config.step_per_epoch = len(train_dataset) // Config.batch_size

# Set up the model
model = DDDNet(Config)

# Train the model
model = train_model(model, train_dataloader, val_dataloader, test_dataloader)

print("Training Done!")