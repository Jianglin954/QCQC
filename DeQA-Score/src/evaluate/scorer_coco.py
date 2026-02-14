from PIL import Image
import torch.nn as nn
import torch
from typing import List
from src.model.builder import load_pretrained_model
from src.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from src.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(*parts):
    return os.path.abspath(os.path.join(BASE_DIR, *parts))


class Scorer(nn.Module):
    def __init__(self, pretrained="zhiyuanyou/DeQA-Score-Mix3", device="cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device)
        prompt = "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"

        self.preferential_ids_ = [id_[1] for id_ in tokenizer(["excellent","good","fair","poor","bad"])["input_ids"]]
        self.weight_tensor = torch.Tensor([5.,4.,3.,2.,1.]).half().to(model.device)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def forward(self, image: List[Image.Image]):
        image = [self.expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in image]
        with torch.inference_mode():
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            output_logits = self.model(
                        input_ids=self.input_ids.repeat(image_tensor.shape[0], 1),
                        images=image_tensor
                    )["logits"][:,-1, self.preferential_ids_]

            return torch.softmax(output_logits, -1) @ self.weight_tensor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="zhiyuanyou/DeQA-Score-Mix3")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_path", type=str, default="fig/singapore_flyer.jpg")
    args = parser.parse_args()

    scorer = Scorer(pretrained=args.model_path, device=args.device)
    
    from PIL import Image, ImageFile
    from pycocotools.coco import COCO
    from tqdm import tqdm
    import os
    
    data_IQA = {
        "captions": [],
        "IQAs": [], 
        "image_ids": []
    }
    
    
    ANN_PATH = resolve_path('../../../', 'coco_data', 'annotations', 'captions_train2017.json')
    IMG_DIR = resolve_path('../../../', 'coco_data', 'train2017')


    coco = COCO(ANN_PATH)
    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, file_name)
        
        IQA_score = scorer([Image.open(img_path).convert("RGB")])
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            data_IQA["captions"].append(ann["caption"].strip())
            data_IQA["image_ids"].append(img_id)
            data_IQA["IQAs"].append(IQA_score.detach().cpu().item())
            caption = ann["caption"].strip()


save_path = resolve_path('../../../', 'processed_data', 'coco', 'data_IQA.pt')  
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(data_IQA, save_path)
