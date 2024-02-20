import os
import torch

# Dataset Configurations
batch_size = 64
image_width = 192
image_size = (3,192,192)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
random_state =7 
num_workers = 4
dataset_name = "oxford_pets" #"oxford_flowers"
dataset_root = "path_to_dataset" 
# Number of plain images to add to the dataset
num_black_images = 500
num_white_images = 500

# VAE Configurations
channels = 64
latent_channels = 64
channel_in = 3

blocks = (1, 2, 4, 8)

lr_vae = 5e-4
nepochs_vae = 150
start_epoch = 0
test_size_vae = 0.05
model_name = f"VAE_Perceptual_{dataset_name}_{image_width}" 
save_dir = os.getcwd()
load_checkpoint  = True
use_cuda = torch.cuda.is_available()
gpu_indx = 0
device = torch.device(gpu_indx if use_cuda else "cpu")


betas=(0.5, 0.999) # for VAE optimization

# Classifier Configurations
n_epochs_classifier = 10
n_epochs_finetuning = 7 # or 3,5 
lr_classifier = 0.001
lr_finetuning = 0.0001
num_classes =  37 if dataset_name == 'oxford_pets' else 112 # 112 for oxford flowers dataset
test_size_clf = 0.3
weight_decay = 1e-5
backbone_type = "resnet" # "vgg16", or "inceptionv1"

# Geodesic Calculation Configurations
num_interpolants = 20 
default_alpha = 0.00005
line_search_alpha_start = 0.1
beta_line_search = 0.5
max_iterations = 300
epsilon = 1000
beta = 0.5
c = 0.001

# Targeted Attributional Attack Configurations
lr_attack = 0.0002
expl_loss_prefactor = 1e17
class_loss_prefactor = 1e06
num_iterations_att = 3000