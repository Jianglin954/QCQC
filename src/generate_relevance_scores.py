import os
import torch
import pickle
import sys
import argparse

from utils import load_index, get_text_feature, get_faiss_sim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(*parts):
    return os.path.abspath(os.path.join(BASE_DIR, *parts))

from search_preparation import index_configs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dataset-condition query completion")
    parser.add_argument('--texts', nargs='+', default=['coco'], help='Choices of captions')
    parser.add_argument('--search_k', type=int, default=1)
    args = parser.parse_args()

    model_id = 'CLIP-Huge-Flickr-Flat'

    out_dir = resolve_path('../', 'coco_faiss_indexes')
    index_dir = os.path.join(out_dir, model_id)
    
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    
    index, feat_transform, img_ids = load_index(index_dir)
    
    ai_config = index_configs[model_id]['a1_config']
    weight_path = index_configs[model_id]['weight_path']

    text_dir = resolve_path('../', 'processed_data', 'coco')

    save_text = True  

    loaded_data = torch.load(os.path.join(text_dir, 'data_aes.pt'))
    loaded_data2 = torch.load(os.path.join(text_dir, 'data_IQA.pt'))
    
    # consistency check
    if loaded_data['captions'] != loaded_data2['captions'] or \
       loaded_data['image_ids'] != loaded_data2['image_ids']:
        print("Two datasets are not aligned. Please check preprocessing.")
        sys.exit(1)

    
    texts = loaded_data['captions']
    image_ids = loaded_data['image_ids']
    
    # extract text features
    q_feats = get_text_feature(texts, ai_config, weight_path)
    q_feats = feat_transform(q_feats)

    faiss_smi, img_hash_list = get_faiss_sim(args.search_k, index, q_feats, img_ids, use_gpu=True)

    if save_text:
        data ={
            "texts": texts,
            "faiss_sim": faiss_smi,
            "faiss_img": img_hash_list,
            "aesthetics": loaded_data['aesthetics'],
            "image_ids": image_ids,
            "IQAs": loaded_data2['IQAs']
        }
        
        torch.save(data, os.path.join(text_dir, 'data.pt'))
        print(f"data saved in {text_dir}!")