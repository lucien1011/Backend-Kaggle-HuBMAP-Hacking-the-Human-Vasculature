import cv2
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from sklearn.model_selection import train_test_split

from backend.augmentation import get_preprocessing
from backend.dataset import HuBMAPDataset,prepare_image_map_annotation
from backend.displayer import Displayer
from backend.utils.io import import_configuration
from backend.utils.seed import seed_everything 

config_path = 'config.20230613_train_v01'
config = import_configuration(config_path)

base_dir = config.base_dir
tag = config.tag
version = config.version
seed = config.seed
model_class = config.model_class
model_args = config.model_args
encoder = config.encoder
encoder_weights = config.encoder_weights
classes = config.classes
pr_threshold = config.pr_threshold

model = model_class(**model_args)
preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
image_map,annotations,golden_ids,silver_ids = prepare_image_map_annotation(base_dir)

valid_image_keys = pickle.load(open(os.path.join(tag,version+'/','valid_image_map.pkl'),'rb'))
valid_image_map = {k:image_map[k] for k in valid_image_keys}
dataset = HuBMAPDataset(valid_image_map,annotations,classes,preprocessing=get_preprocessing(preprocessing_fn))
model.load_state_dict(torch.load(os.path.join(tag,version+'/','best_model.pth'),map_location=torch.device('cpu')))

displayer = Displayer(valid_image_map,annotations,classes)
output_dir = os.path.join(tag,version+'/','display/')
os.makedirs(output_dir,exist_ok=True)
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        key,image,_ = dataset[i]
        image = torch.tensor(image).reshape((1,*image.shape))
        mask = model(image) > pr_threshold
        displayer.display_image(key,{'blood_vessel':mask[0,0,:],},os.path.join(output_dir,'display_{:d}.png'.format(i)))
