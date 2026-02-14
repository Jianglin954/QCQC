import glob
import os
import PIL
import PIL.Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip


class SingleFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(folder_path, "*"))
        print('Found {} images in {}'.format(len(self.image_paths), folder_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = PIL.Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)


def extract_feats(index_config):

    ai_config = index_config['a1_config']
    weight_path = index_config['weight_path']
    img_dir = index_config['img_dir']
    batch_size = 1024 # 64

    model, _, transform = open_clip.create_model_and_transforms(ai_config, pretrained=weight_path)

    devive = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', devive)
    model = model.to(devive)

    dataset = SingleFolderDataset(img_dir, transform=transform)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    im_ids = []
    im_feats = []

    for i, (patched_tensor, img_id) in tqdm(enumerate(dl)):
        patched_tensor = patched_tensor.to(devive)

        with torch.no_grad():
            out = model.encode_image(patched_tensor)

        im_ids.append(img_id)
        im_feats.append(out.cpu().numpy())

    im_hashes = np.concatenate(im_ids)
    im_feats = np.concatenate(im_feats)
    return im_hashes, im_feats


