<div align="center">
<h1> Do You Act Like You Talk? Exploring Pose-based Driver Action Classification with Speech Recognition Networks </h1>


[Pablo Pardo-Decimavilla](https://www.linkedin.com/in/pablo-pardo-decimavilla/)</a><sup><span>1*</span></sup>, 
Luis M. Bergasa</a><sup><span>1</span></sup>,
Santiago Montiel-Marín</a><sup><span>1</span></sup>,
Miguel Antunes</a><sup><span>1</span></sup>,
Ángel Llamazares</a><sup><span>1</span></sup>,
</br>

<sup>1</sup> Electronics Departament, University of Alcalá (UAH), Spain.
<sup>*</sup> Corresponding author.
<br>
<div>

<img src="images/input_example_.jpg" width=60%>

</div>
</div>

# Replicate paper results

Follow the steps to replicate the paper results.

## Run the docker container:

Build and run the docker container to replicate the same experimental setup.

> Requires docker engine and nvidia docker for GPU training and evaluation.


```bash
cd docker
docker compose build
docker compose run --rm dyalyt_container
```

## Prepare the dataset

To train the model on [**Drive&Act**](https://driveandact.com/):

The following commands will download the the [3D Body Pose](https://driveandact.com/dataset/iccv_openpose_3d.zip) and the [Activities annotations](https://driveandact.com/dataset/iccv_activities_3s.zip) from the [official site](https://driveandact.com/). The data is then processed:

```bash
cd driveandact
wget https://driveandact.com/dataset/iccv_openpose_3d.zip
wget https://driveandact.com/dataset/iccv_activities_3s.zip
unzip iccv_openpose_3d.zip
unzip iccv_activities_3s.zip
python3 prepare_aad.py --data --coarse_label
cd ..
```
> The script is based on [this repo](https://github.com/holzbock/st_mlp).<br>

This will create two pickle file one containing the skeleton (```skeleton.pkl```) and the other with the corresponding annotations (```coarse.pkl```).


## Training and Testing

The following command will train the model with the hyperparameters used in the paper. After the training the the best model will be evaluated in the test set.
> Note that results may differ slightly when training in another architecture. 

### Logging with Wandb

You will be asked to add your wandb token. It will be saved in the Project named ```DYALYT``` with the specified ```run-name```.

```bash
python3 train.py --wandb <run-name>
```

### Not logging

If you do not want to log 
```bash
python3 train.py
```

## Testing
Download the weights from [here](https://drive.google.com/file/d/1Ql0u7Kc5vSoasSXHA1b4z9y5OCyjqf6C/view?usp=sharing).<br>
The following command will load the pre-trained weights and test it.
```bash
python3 test.py --weights dyalyt_coarse_0.4359.ckpt
```

# Results

Evaluation of coarse scenarios/tasks on the Drive\&Act dataset using macro-accuracy of our architecture.
> Trained in a NVIDIA GeForce RTX 2080 Ti 

| Data | Method | Validation | Test  |
|---------------|-----------------|---------------------|----------------|
| Pose    | ours   | 44.60      | 43.59 |

# Contact
To contact us, please write an email to: pablo.pardod@uah.es
