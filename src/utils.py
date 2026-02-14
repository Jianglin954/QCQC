import torch
import os
import numpy as np
import faiss
import open_clip
import functools
import re
from tqdm import tqdm
import ipdb

from torch.utils.data import DataLoader


def contains_special_characters(text):
    # check if non-ASCII characters exist
    if re.search(r'[^\x00-\x7F]', text):
        return True
    return False

def check_texts_for_special_characters(texts):
    results = []
    for i, text in enumerate(texts):
        if contains_special_characters(text):
            results.append(f"Text {i}: Contains special characters")
    return results

def clean_text(text):
    # remove non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)  
    # remove redundent space
    text = re.sub(r'\s+', ' ', text)  
    # remove space at the beginning and end of texts
    text = text.strip()
    return text

def clean_texts(texts):
    return [clean_text(text) for text in texts]




def load_ori_query(coco_class_path):
    with open(coco_class_path, 'r') as file:
        coco_classes = [line.strip() for line in file.readlines()] 

    def add_article_to_classes(class_list):
        result = []
        for item in class_list:
            # Check if the first letter of the item is a vowel (a, e, i, o, u)
            if item[0].lower() in 'aeiou':
                result.append(f"an {item}")
            else:
                result.append(f"a {item}")
        return result

    a_cls_list = add_article_to_classes(coco_classes)

    an_image_showing_list = [f"an image showing {cls}" for cls in coco_classes]

    return a_cls_list, an_image_showing_list



def load_index(index_dir):
    print(os.getcwd())
    index_path = os.path.join(index_dir, 'faiss_IVPQ_PCA.index')
    index = faiss.read_index(index_path)

    # Load the transformations
    norm1 = faiss.read_VectorTransform(os.path.join(index_dir, 'norm1.bin'))
    do_pca = os.path.exists(os.path.join(index_dir, 'pca.bin'))
    if do_pca:
        pca = faiss.read_VectorTransform(os.path.join(index_dir, 'pca.bin'))
        norm2 = faiss.read_VectorTransform(os.path.join(index_dir, 'norm2.bin'))

    def feat_transform(x):
        x = norm1.apply_py(x)
        if do_pca:
            x = pca.apply_py(x)
            x = norm2.apply_py(x)
        return x

    img_ids = np.load(os.path.join(index_dir, 'img_ids.npy'))

    return index, feat_transform, img_ids


def load_model(config_name, weight_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms(config_name, pretrained=weight_path)
    tokenizer = open_clip.get_tokenizer(config_name)

    if device == 'cpu':
        model = model.float().to(device) # CPU does not support half precision operations
    else:
        model = model.to(device)
    model.eval()
    return model, tokenizer




def get_text_list_feature(query_list, ai_config, weight_path):
    '''
    query_list: n classes, each class has k queries !
    '''
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(ai_config, weight_path)

    
    text_list = [tokenizer(query).to(device) for query in query_list]

    with torch.no_grad():
        text_feats = [model.encode_text(text) for text in text_list]

    text_feats = [text.cpu().numpy() for text in text_feats]
    return text_feats




def get_text_feature(query_list, ai_config, weight_path):
    '''
    query_list: n queries !
    '''
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(ai_config, weight_path)

    text_list = tokenizer(query_list).to(device) 

    num = text_list.shape[0]
    batch_size = 1000 # 5000  

    with torch.no_grad():   
        text_feats = []
        for i in tqdm(range(0, num, batch_size)):
            text_feats.append(model.encode_text(text_list[i:i + batch_size]))
        #text_feats = model.encode_text(text_list) 
        text_feats = torch.cat(text_feats, dim=0)

    del model
    torch.cuda.empty_cache()

    return text_feats.cpu().numpy() 




def print_scores(aesthetics, faiss_smi):
    # np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
    aesthetics = np.array(aesthetics)
    average_aesthetics = np.around(np.mean(aesthetics, axis=0), decimals=3)

    faiss_smi = np.array(faiss_smi)
    average_similarities = np.around(np.mean(faiss_smi, axis=0), decimals=3)

    avg_aes, std_aes = np.mean(aesthetics), np.std(aesthetics)
    avg_smi, std_smi = np.mean(faiss_smi), np.std(faiss_smi)
    
    print("avg aesthetics for each completion:", ' '.join(map(str, average_aesthetics)))
    print("avg aesthetics over all images: {:.3f}".format(avg_aes))
    print("std aesthetics over all images: {:.3f}".format(std_aes))
    print("avg similarities for each completion:", ' '.join(map(str, average_similarities)))
    print("avg similarities over all images: {:.3f}".format(avg_smi))
    print("std similarities over all images: {:.3f}".format(std_smi))
    print("---------------------------------------------------------------------------")



def print_scores_iqa(aesthetics, faiss_smi, iqas):
    # np.set_printoptions(precision=3, suppress=False, floatmode='fixed')
    aesthetics = np.array(aesthetics)
    average_aesthetics = np.around(np.mean(aesthetics, axis=0), decimals=3)

    faiss_smi = np.array(faiss_smi)
    average_similarities = np.around(np.mean(faiss_smi, axis=0), decimals=3)

    iqas = np.array(iqas)
    average_iqas = np.around(np.mean(iqas, axis=0), decimals=3)
    
    
    avg_aes, std_aes = np.mean(aesthetics), np.std(aesthetics)
    avg_smi, std_smi = np.mean(faiss_smi), np.std(faiss_smi)
    avg_iqa, std_iqa = np.mean(iqas), np.std(iqas)
    
    print("avg aesthetics for each completion:", ' '.join(map(str, average_aesthetics)))
    print("avg aesthetics over all images: {:.3f}".format(avg_aes))
    print("std aesthetics over all images: {:.3f}".format(std_aes))
    print("avg similarities for each completion:", ' '.join(map(str, average_similarities)))
    print("avg similarities over all images: {:.3f}".format(avg_smi))
    print("std similarities over all images: {:.3f}".format(std_smi))
    
    print("avg IQA for each completion:", ' '.join(map(str, average_iqas)))
    print("avg IQA over all images: {:.3f}".format(avg_iqa))
    print("std IQA over all images: {:.3f}".format(std_iqa))
    print("---------------------------------------------------------------------------")



def get_scores(img_list, dis_list, loaded_data, img_ids):

    aesthetics_score = loaded_data["aesthetics_score"]
    strImagehash = loaded_data["strImagehash"]

    img_hash_list = []     
    for imgs in img_list:          
        img_hash = [[img_ids[idx] for idx in img] for img in imgs]   #  imgs: [10, 100], img: [100]
        img_hash_list.append(img_hash)      

    aesthetics = []
    for each_class in img_hash_list:   # for each class in 80 classes
        avg_aesthetic = []
        for each_completion in each_class:   # for each completion in 10 completions
            aes_score = []
            # img_hash_set = set(each_completion)    # 100 retrieved images
            # indices = [i for i, hash_str in enumerate(strImagehash) if hash_str in img_hash_set]

            indices = [strImagehash.index(s) if s in strImagehash else None for s in each_completion]
            aes_score = [aesthetics_score[iii] if iii is not None else aesthetics_score.mean() for iii in indices]
            # torch.tensor(4.9504)    aesthetics_score.mean()
            aes_score = torch.stack(aes_score)
            
            avg_aesthetic.append(aes_score.mean())
        aesthetics.append(torch.stack(avg_aesthetic))
    aesthetics = torch.stack(aesthetics)

    faiss_smi = [[each_completion.mean() for each_completion in each_class] for each_class in dis_list]
    faiss_smi = torch.tensor(faiss_smi)     # faiss_smi: [80, 10]

    return aesthetics, faiss_smi, img_hash_list




def get_scores_prompt(img_list, dis_list, loaded_data, img_ids):

    aesthetics_score = loaded_data["aesthetics_score"]
    strImagehash = loaded_data["strImagehash"]

    img_hash_list = []     
    for imgs in img_list:          
        img_hash = [[img_ids[idx] for idx in img] for img in imgs]   #  imgs: [10, 100], img: [100]
        img_hash_list.append(img_hash)      

    aesthetics_all = []
    for each_class in img_hash_list:   # for each class in 80 classes
        aesthetic = []
        for each_completion in each_class:   # for each completion in 10 completions
            aes_score = []
            # img_hash_set = set(each_completion)    # 100 retrieved images
            # indices = [i for i, hash_str in enumerate(strImagehash) if hash_str in img_hash_set]

            indices = [strImagehash.index(s) if s in strImagehash else None for s in each_completion]
            aes_score = [aesthetics_score[iii] if iii is not None else aesthetics_score.mean() for iii in indices]
            # torch.tensor(4.9504)    aesthetics_score.mean()
            aes_score = torch.stack(aes_score)
            
            aesthetic.append(aes_score)
        aesthetics_all.append(torch.stack(aesthetic))
    aesthetics_all = torch.stack(aesthetics_all)

    faiss_smi = [[each_completion for each_completion in each_class] for each_class in dis_list]
    faiss_smi = torch.tensor(faiss_smi)     # faiss_smi: [80, 10]

    return aesthetics_all, faiss_smi



def image_retrive(sear_k, index, q_feats, loaded_data, img_ids):

    img_list = []
    dis_list = []
    for q_feat in q_feats:
        D, I = index.search(q_feat, sear_k)         # D, I: [10, 100]
        img_list.append(I)          # img_list {[10, 100], [10, 100], [10, 100]}, len(80)
        dis_list.append(D)          # dis_list {[10, 100], [10, 100], [10, 100]}, len(80)

    aesthetics, faiss_smi, img_hash_list = get_scores(img_list, dis_list, loaded_data, img_ids)
    # ipdb.set_trace()
    
    print_scores(aesthetics, faiss_smi)
    return img_hash_list, dis_list





def image_retrive_prompt(sear_k, index, q_feats, loaded_data, img_ids):
    img_list = []
    dis_list = []
    for q_feat in q_feats:
        D, I = index.search(q_feat, sear_k)         # D, I: [10, 100]
        img_list.append(I)          # img_list {[10, 100], [10, 100], [10, 100]}, len(80)
        dis_list.append(D)          # dis_list {[10, 100], [10, 100], [10, 100]}, len(80)
    ipdb.set_trace()

    aesthetics, faiss_smi = get_scores_prompt(img_list, dis_list, loaded_data, img_ids)
    return aesthetics.squeeze().squeeze(), faiss_smi.squeeze().squeeze()



def get_faiss_sim(sear_k, index, q_feats, img_ids, use_gpu):

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

        
        num = q_feats.shape[0]
        batch_size = 100000 # 1000000
        
        img_hash_list = []
        faiss_smi = []

        for i in tqdm(range(0, num, batch_size)):
            D, I = index.search(q_feats[i:i + batch_size], sear_k)   
            img_hash_list.append(img_ids[I.squeeze()])
            faiss_smi.append(torch.from_numpy(D.squeeze()))

        faiss_smi = torch.cat(faiss_smi, dim=0)
        
        return faiss_smi, img_hash_list

    D, I = index.search(q_feats, sear_k) 
    img_hash_list = img_ids[I.squeeze()]        
    faiss_smi = torch.from_numpy(D.squeeze())  
    
    return faiss_smi, img_hash_list
