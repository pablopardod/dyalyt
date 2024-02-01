
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import pandas as pd

## Torchvision
from torchmetrics.classification import MulticlassAccuracy

# PyTorch Lightning
import pytorch_lightning as pl

# Confussion Matrix
from torchmetrics import  ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

# External Modules
from models.model import Net

#Wandb
import wandb

#For Cutmix
from torchvision.transforms import v2
from augmentations import CutMixModified

class_names = ['final_task', 'work', 'eat_drink', 'read_write_magazine', 'put_on_jacket', 'take_off_sunglasses', 
                                'read_write_newspaper', 'fasten_seat_belt', 'put_on_sunglasses', 'watch_video', 'take_off_jacket', 
                                'hand_over']                           

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class DDDNet(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = Net(cfg).to(cfg.DEVICE)
        self.cfg = cfg
        self.preds_val = []
        self.batch_val = []

        # Extra val_logs
        self.mca = MulticlassAccuracy(num_classes=cfg.num_classes, average=None)
        self.amicro = MulticlassAccuracy(num_classes=cfg.num_classes,average='micro')
        self.amacro = MulticlassAccuracy(num_classes=cfg.num_classes,average='macro')
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=cfg.num_classes)
        self.class_names = class_names
        
        self.train_loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.val_loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, x, mode):
        x['input'] = x['input'].permute(0, 3, 1, 2)
        if mode == "train":
            # CutMix modified
            cutmix_original = CutMixModified(num_classes=self.cfg.num_classes, mode='original')
            cutmix_horizontal = CutMixModified(num_classes=self.cfg.num_classes, mode='horizontal')
            cutmix_vertical = CutMixModified(num_classes=self.cfg.num_classes, mode='vertical')
            cutmix = v2.RandomChoice([cutmix_horizontal, cutmix_vertical, cutmix_original])
           
            x['input'], x['action'] = cutmix(x['input'], x['action'])

        return self.model(x)        


    def configure_optimizers(self):
        
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup= (self.cfg.total_training_steps * self.cfg.warmup) // 100, #in batches
                                             max_iters=self.cfg.total_training_steps)
        

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]


    def _calculate_loss(self, batch, mode="train"):
        preds = self.forward(batch, mode=mode)

        #Loss
        if mode == "train":
            loss = self.train_loss_function(preds, batch['action'])
        else:
            loss = self.val_loss_function(preds, batch['action'])

        self.log(f'{mode}_loss', loss)

        if not mode == "train":
            # Append preds and batch
            self.preds_val.append(preds)
            self.batch_val.append(batch['action'])

            if mode == "val":
                # Update confussion matrix
                self.confmat.update(preds, batch['action'])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")
        
    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
    
    def _log_confussion_matrix(self):
        confmat = self.confmat.compute()
        class_names = self.class_names
        df_cm = pd.DataFrame(confmat.cpu().numpy() , index = [i for i in class_names], columns = [i for i in class_names])
      
        # Normalise the confusion matrix 
        norm =  np.sum(df_cm, axis=1)
        normalized_cm = (df_cm.T/norm).T # 
        
        f, ax = plt.subplots(figsize = (15,10)) 
        sn.heatmap(normalized_cm, annot=True, ax=ax)
        plt.savefig('heatmap.jpg')  

        self.logger.log_image(key=f"cm_epoch_{self.current_epoch}", images=["heatmap.jpg"])

        self.confmat.reset()  #This was NEEDED otherwise the confusion matrix kept stacking the results after each epoch 

    # To calculate 'best_val_loss'
    def on_validation_epoch_end(self) -> None:
        best_val_loss = self.trainer.callback_metrics.get('best_val_loss')
        best_val_macro = self.trainer.callback_metrics.get('best_val_macro')

        preds_val = torch.cat(self.preds_val, dim=0)
        batch_val = torch.cat(self.batch_val, dim=0)

        loss = F.cross_entropy(preds_val, batch_val)

        macro = self.amacro(preds_val, batch_val)
        micro = self.amicro(preds_val, batch_val) 

        self.log('val_acc_micro', micro)
        self.log('val_acc_macro', macro)

        if best_val_loss is None or loss < best_val_loss:
            self.log('best_val_loss', loss, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        if best_val_macro is None or macro > best_val_macro:
            self.log('best_val_macro', macro, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        self.preds_val = []
        self.batch_val = []
        
        #Log acc_val per class 
        if self.logger: 
            accs = self.mca(preds_val, batch_val).to('cpu').tolist()
            self.logger.log_table(key=f"table_epoch_{self.current_epoch}", columns=class_names, data=[accs])

            #Log a confussion matrix
            self._log_confussion_matrix()

        return super().on_validation_epoch_end()
    
    def on_test_epoch_end(self) -> None:
        preds_val = torch.cat(self.preds_val, dim=0)
        batch_val = torch.cat(self.batch_val, dim=0)

        loss = F.cross_entropy(preds_val, batch_val)

        macro = self.amacro(preds_val, batch_val)
        micro = self.amicro(preds_val, batch_val) 

        self.log('test_acc_micro', micro)
        self.log('test_acc_macro', macro)

        if self.logger: 
            accs = self.mca(preds_val, batch_val).to('cpu').tolist()
            self.logger.log_table(key=f"acc_test_per_class", columns=class_names, data=[accs])


        self.preds_val = []
        self.batch_val = []

        return super().on_test_epoch_end()

    def finish_run(self):
        wandb.finish()