import cv2
import glob
import json
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as BaseDataset

def prepare_image_map_annotation(base_dir):
    
    annotations = {}
    with open(os.path.join(base_dir,'polygons.jsonl'), 'r') as f:
        for line in f:
            annotation = json.loads(line)
            image_id = annotation['id']
            image_annotations = annotation['annotations']
            annotations[image_id] = image_annotations
    
    train = glob.glob(os.path.join(base_dir,"train/*"))
    test = glob.glob(os.path.join(base_dir,"test/*"))
    image_map = {}
    for impath in train:
        key = impath.split('/')[-1].split('.')[0]
        image_map[key] = impath

    df_msi = pd.read_csv(os.path.join(base_dir,'wsi_meta.csv'))
    df_tile = pd.read_csv(os.path.join(base_dir,'tile_meta.csv'))
    golden_ids = df_tile.loc[df_tile.dataset==1,'id'].tolist()
    silver_ids = df_tile.loc[df_tile.dataset==2,'id'].tolist()

    return image_map,annotations,golden_ids,silver_ids

class HuBMAPDataset(BaseDataset):
    CLASSES = ['blood_vessel', 'glomerulus', 'unsure']
    
    def __init__(
            self, 
            image_map, 
            annotations=None, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):

        self.image_map = image_map
        self.image_keys = list(self.image_map.keys())
        self.annotations = annotations

        self.classes = self.CLASSES if classes is None else classes
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        key = self.image_keys[i]

        image = cv2.imread(self.image_map[key])
        
        if self.annotations:
            if key in self.annotations:
                masks = {cls: np.zeros((512,512), dtype=np.uint8) for cls in self.classes}
                polygons = self.annotations[key]
                for polygon in polygons:
                    annotation_type = polygon['type']
                    if annotation_type not in self.classes: continue
                    lines = np.array(polygon['coordinates'])
                    lines = lines.reshape(-1, 1, 2)
                    cv2.fillPoly(masks[annotation_type], [lines], 1)
                mask = np.stack([masks[cls] for cls in self.classes], axis=-1).astype('float')
            else:
                mask = np.zeros((512, 512, len(self.classes)), dtype=float)
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            if self.annotations:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                sample = self.preprocessing(image=image)
                image = sample['image']
 
        if self.annotations:
            return key, image, mask
        else:
            return key, image
 
    def __len__(self):
        return len(self.image_keys)

