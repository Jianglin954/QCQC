import webdataset as wds
from PIL import Image
import io
# import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip

from PIL import Image, ImageFile
from pycocotools.coco import COCO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(*parts):
    return os.path.abspath(os.path.join(BASE_DIR, *parts))


ANN_PATH = resolve_path('..', 'coco_data', 'annotations', 'captions_train2017.json')
IMG_DIR = resolve_path('..', 'coco_data', 'train2017')                              # image folder


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

model_path = resolve_path('.', 'sac+logos+ava1-l14-linearMSE.pth') 
s = torch.load(model_path)   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   

data = {
    "captions": [],
    "aesthetics": [], 
    "image_ids": []
}


coco = COCO(ANN_PATH)
img_ids = coco.getImgIds()

for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]
    img_path = os.path.join(IMG_DIR, file_name) 
        
    pil_image = Image.open(img_path)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy() )

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))



    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        data["captions"].append(ann["caption"].strip())
        data["image_ids"].append(img_id)
        data["aesthetics"].append(prediction.detach().cpu().item())
        caption = ann["caption"].strip()

save_path = resolve_path('..', 'processed_data', 'coco', 'data_aes.pt')  
os.makedirs(os.path.dirname(save_path), exist_ok=True) 

torch.save(data, save_path)