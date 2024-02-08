import time
import wandb 
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore")
import augmentations as A

#Argparse
import argparse


def test_model(model,test_loader, checkpoint_path):

    # create run_id
    run_id = str(int(time.time()))

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

    trainer = pl.Trainer(#default_root_dir=os.path.join(Config.CHECKPOINT_PATH),
                         default_root_dir=checkpoint_path,
                         accelerator="gpu" if str(Config.DEVICE).startswith("cuda") else "cpu",
                         devices=1,
                         logger=wandb_logger,
                         deterministic=True)


    print(f"Loading run: {checkpoint_path}")
    model = model.load_from_checkpoint(checkpoint_path)
    trainer.test(model, test_loader)

    model.finish_run()
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", type=str, default='False', help="log with wandb. 'False' to disable")
parser.add_argument("--weights", type=str, help="pre-trained weights path")
parser.add_argument('--split', type=int, default=0, choices = [0,1,2])
parser.add_argument('--config', type=str, default='cfg2aad', help='select config (model) for training (cfg1 or cfg2)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')

#Dropout
parser.add_argument('--input_dropout_p', type=float, default=0.2484335126263782, help='input_dropout_p')
parser.add_argument('--feed_forward_dropout_p', type=float, default=0.16543905095482592, help='feed_forward_dropout_p')
parser.add_argument('--attention_dropout_p', type=float, default=0.25975470832480363, help='attention_dropout_p')
parser.add_argument('--conv_dropout_p', type=float, default=0.1855344530469696, help='conv_dropout_p')

#Model parameters
parser.add_argument('--input_dim', type=int, default=120, help='input_dim') #Not used
parser.add_argument('--encoder_dim', type=int, default=120, help='encoder_dim')
parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
parser.add_argument('--num_attention_heads', type=int, default=4, help='num_attention_heads')


args = parser.parse_args()

#Load dataset
from data.driveandact import CustomDataset
# Load trainer
from models.trainer import DDDNet
# Load config
from configs.cfg import Config 


#Apply args
Config.weights = args.weights
Config.split = args.split
Config.batch_size = args.batch_size 

Config.input_dropout_p = args.input_dropout_p
Config.feed_forward_dropout_p = args.feed_forward_dropout_p
Config.attention_dropout_p = args.attention_dropout_p
Config.conv_dropout_p = args.conv_dropout_p

Config.input_dim = args.encoder_dim
Config.encoder_dim = args.encoder_dim 
Config.num_layers = args.num_layers
Config.num_attention_heads = args.num_attention_heads

print(f"Using {'GPU' if 'cuda' in str(Config.DEVICE) else 'CPU'}")

print('ℹ️  Label type: ',Config.label_type_aad)
print('ℹ️  Dataset: ',Config.dataset)


# Set up the dataset and dataloader 
path = 'skeletons.pkl'
test_dataset = CustomDataset(Config.data_path, mode='test', split=Config.split, label_type=Config.label_type_aad, 
                                seq_len=Config.MAX_LEN, num_classes=Config.num_classes, convert_coord=Config.convert_coord, n_landmarks=Config.n_landmarks, path=path)

test_dataloader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=None,
    )

# Set up the model
model = DDDNet(Config)

# Train the model
model = test_model(model, test_dataloader, Config.weights)

print("Testing Done!")