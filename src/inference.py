import os
import sys
import argparse
import numpy as np
import torch
from utils import *
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, GPT2Tokenizer, GPT2LMHeadModel
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from search_preparation import index_configs


def resolve_path(*parts):
    return os.path.abspath(os.path.join(BASE_DIR, *parts))


def load_index_f(index_dir):
    print(os.getcwd())
    index_path = os.path.join(index_dir, 'faiss_IVPQ_PCA.index')
    index = faiss.read_index(index_path)

    norm1 = faiss.read_VectorTransform(os.path.join(index_dir, "norm1.bin"))
    do_pca = os.path.exists(os.path.join(index_dir, "pca.bin"))
    if do_pca:
        pca = faiss.read_VectorTransform(os.path.join(index_dir, "pca.bin"))
        norm2 = faiss.read_VectorTransform(os.path.join(index_dir, "norm2.bin"))

    def feat_transform(x):
        x = norm1.apply_py(x)
        if do_pca:
            x = pca.apply_py(x)
            x = norm2.apply_py(x)
        return x

    img_ids = torch.load(os.path.join(index_dir, 'img_ids.pt'), weights_only=False)

    return index, feat_transform, img_ids

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stops = [], encounters=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        current_token = input_ids[0]
        for stop in self.stops:
            if self.tokenizer.decode(stop) in self.tokenizer.decode(current_token):
                return True
        return False

def query_completion(model_name, ori_query, args, text_dir):

    if model_name == "gpt2" or model_name == "Qwen":
        model_save_path = model_name
    elif model_name == 'no_completion':
        print(f"No query completion, copy original query {args.cmpl_k} times to keep the shape ! ")
        return [[item] * args.cmpl_k for item in ori_query]       # copy 10 times to keep the same shape with the ones using k auto-completions 
    else:
        # Path: outputs/gpt2coco{suffix}/model_{suffix}/checkpoint, e.g. model_name="model_ADS" -> gpt2cocoADS/model_ADS/checkpoint
        suffix = model_name.replace("model_", "", 1) if model_name.startswith("model_") else model_name
        if "Qwen" in model_name:
            tmp_path = os.path.join(text_dir, f"qwencoco{suffix}", model_name)
        else:
            tmp_path = os.path.join(text_dir, f"gpt2coco{suffix}", model_name)
        model_save_path = os.path.join(tmp_path, "checkpoint")
    print(f"loading model from {model_save_path}...")
    
    if model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
        model = GPT2LMHeadModel.from_pretrained(model_save_path)
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name == "Qwen":
        model = AutoModelForCausalLM.from_pretrained(model_save_path+"/Qwen2.5-0.5B")  
        tokenizer = AutoTokenizer.from_pretrained(model_save_path+"/Qwen2.5-0.5B")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        if "Qwen" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_save_path)
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        else: 
            tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
            model = GPT2LMHeadModel.from_pretrained(model_save_path)

    #tokenizer.pad_token = tokenizer.eos_token  # This explicitly sets the pad token to the same value as the EOS token, ensuring that padding is handled.
    #tokenizer.padding_side ='left'
    #model.config.pad_token_id = tokenizer.eos_token_id
    #model.config.eos_token_id = tokenizer.eos_token_id 
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")



    stop_words = [".", "!", "?"]   
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=stop_words_ids)])


    if args.no_condition:
        with torch.no_grad():  
            autocompleted_queries = []
            for _ in range(80):
                generated_ids = model.generate(
                    do_sample = True, 
                    min_length = args.min_len, 
                    #max_length = args.max_len,
                    max_new_tokens = args.max_len, 
                    temperature = args.tmpr,     
                    top_k = args.top_k,
                    top_p = args.top_p,
                    repetition_penalty = args.rept_pnal,
                    no_repeat_ngram_size = args.no_rept_ngram,
                    eos_token_id = tokenizer.eos_token_id ,    # Use the end-of-sentence token as a stop condition
                    pad_token_id = tokenizer.pad_token_id,  # tokenizer.pad_token_id,
                    stopping_criteria = stopping_criteria,
                    num_return_sequences = 1,    
                )  
                autocompleted_queries.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True)) 
            
            autocompleted_queries = [[item] * args.cmpl_k for item in autocompleted_queries]   
            
    else:
        with torch.no_grad():    
            tokenized_inputs = [tokenizer(text, padding=True, truncation=True, return_tensors="pt") for text in ori_query]  # len(tokenized_inputs)=80
            autocompleted_queries = []
            for input_data in tokenized_inputs:
                queries_for_each_class = []
                for i in range(args.cmpl_k):
                    generated_ids = model.generate(
                                            input_data['input_ids'], 
                                            attention_mask = input_data["attention_mask"],      
                                            do_sample = True, 
                                            min_length = args.min_len, 
                                            #max_length = args.max_len,
                                            max_new_tokens = args.max_len, 
                                            temperature = args.tmpr,                        # Control randomness (lower temperature makes the output more deterministic)
                                            top_k = args.top_k,
                                            top_p = args.top_p,
                                            repetition_penalty = args.rept_pnal,
                                            no_repeat_ngram_size = args.no_rept_ngram,
                                            eos_token_id = tokenizer.eos_token_id ,    # Use the end-of-sentence token as a stop condition
                                            pad_token_id = tokenizer.pad_token_id,  # tokenizer.pad_token_id,
                                            stopping_criteria = stopping_criteria,
                                            return_dict_in_generate=False,  # if true: generated_ids[0] should be --> generated_ids["sequences"][0]
                                            num_return_sequences = 1,    # only generate one sentence at a time, stopping_criteria didn't work well for multi-outputs
                                        )  
                    # tokenizer.decode(model.generate(input_data['input_ids'], attention_mask = input_data["attention_mask"], max_new_tokens=200, num_return_sequences = 1, eos_token_id = tokenizer.eos_token_id , pad_token_id = tokenizer.pad_token_id, )[0])
                    queries_for_each_class.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True))             
                autocompleted_queries.append(queries_for_each_class)  
    return autocompleted_queries 


def image_retrieve_coco(sear_k, index, q_feats, loaded_data, img_ids):

    img_list = []
    dis_list = []
    
    D, I = index.search(q_feats, sear_k)        
          
    aesthetics, faiss_smi, iqas, img_hash_list = get_scores_coco(I, D, loaded_data, img_ids)
    
    print_scores_iqa(aesthetics, faiss_smi, iqas)
    return aesthetics, faiss_smi, iqas, img_hash_list


def get_scores_coco(I, D, loaded_data, img_ids):
    aesthetics_score = torch.tensor(loaded_data["aesthetics"])
    IQAs_score = torch.tensor(loaded_data["IQAs"])
    strImagehash = loaded_data["image_ids"]
       
    img_hash = [img_ids[idx] for idx in I]   
    
    aesthetics = []
    iqas = []
    for each_class in img_hash:  
        indices = [strImagehash.index(int(s.split(".")[0])) if int(s.split(".")[0]) in strImagehash else None for s in each_class]
        aes_score = [aesthetics_score[iii] if iii is not None else aesthetics_score.mean() for iii in indices]
        iqa_score = [IQAs_score[iii] if iii is not None else IQAs_score.mean() for iii in indices]

        aes_score = torch.stack(aes_score)
        aesthetics.append(aes_score)
        
        iqa_score = torch.stack(iqa_score)
        iqas.append(iqa_score)
          
    aesthetics = torch.stack(aesthetics)
    iqas = torch.stack(iqas)
    faiss_smi = torch.tensor(D)  

    return aesthetics, faiss_smi, iqas, img_hash



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dataset-condition query completion")

    parser.add_argument('--cmpl_k', type=int, default=10, help='perform k times query completion')
    parser.add_argument('--sear_k', type=int, default=50, help='search k images for each completed query')
    
    parser.add_argument('--min_len', type=int, default=10, help='minimal length in query completion')
    parser.add_argument('--max_len', type=int, default=20, help='maximal length in query completion')
    parser.add_argument('--tmpr', type=float, default=0.7, help='temperature in query completion')
    parser.add_argument('--top_k', type=int, default=50, help='select tok_k tokens in query completion')
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p in query completion')
    parser.add_argument('--rept_pnal', type=float, default=1.2, help='repetition_penalty in query completion')
    parser.add_argument('--no_rept_ngram', type=int, default=2, help='no_repeat_ngram_size in query completion')
    
    parser.add_argument('--seed', type=int, default=42, help='seed')
     
    parser.add_argument('--search_imgs', action='store_true', help='load pre-downloaded data for image retrieval')
    parser.add_argument('--an_img_showing', action='store_true', help='query: an image showing x')
    parser.add_argument('--prt_cmpl_qry', action='store_true', help='print some completed queries for illustration')
    
    parser.add_argument('--no_condition', action='store_true', help='generate texts without conditions')
    parser.add_argument('--model_names', nargs='+', default=['no_completion', 'gpt2'], help='Choices of completion models')

    
    parser.add_argument('--aes_level', type=str, default='high', help='low or high or median')
    parser.add_argument('--sim_level', type=str, default='high', help='low or high or median')
    parser.add_argument('--iqa_level', type=str, default='high', help='low or high or median')
    

    
    args = parser.parse_args()
    set_seed(args.seed)


    
    work_space = resolve_path("..")
    text_dir = resolve_path("..", "outputs")
    coco_class_path = os.path.join(work_space, "MC-COCO-Class.txt")
    if not os.path.exists(coco_class_path):
        raise FileNotFoundError(f"Class list not found: {coco_class_path}")
    a_cls_list, an_image_showing_list = load_ori_query(coco_class_path)


    index_dir = resolve_path("..", "coco_faiss_indexes", "CLIP-Huge-Flickr-Flat")
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Index dir not found: {index_dir}. Run search_preparation.py first.")
    index, feat_transform, img_ids = load_index(index_dir)
    print("loaded index, transform, img_ids.")

    model_id = 'CLIP-Huge-Flickr-Flat'
    ai_config = index_configs[model_id]['a1_config']
    weight_path = index_configs[model_id]['weight_path']


    for model_name in args.model_names:
        print(args)

        data_dict = {'text': np.array(a_cls_list)}
        dataset = Dataset.from_dict(data_dict)

        # Prompt templates: must match training order (A=Aesthetic, D=DeQA, S=Similarity)
        PROMPT_TEMPLATES = {
            # "gpt2": "Similarity: {sim}, Aesthetic: {aes}, Query: {cond}",
            # "Qwen": "Similarity: {sim}, Aesthetic: {aes}, Query: {cond}",
            "model_ADS": "<|startoftext|>Aesthetic: {aes}, DeQA-Score: {iqa}, Similarity: {sim}, Query: {cond}",
            "model_DSA": "<|startoftext|>DeQA-Score: {iqa}, Similarity: {sim}, Aesthetic: {aes}, Query: {cond}",
            "model_SAD": "<|startoftext|>Similarity: {sim}, Aesthetic: {aes}, DeQA-Score: {iqa}, Query: {cond}",
            "model_SA": "<|startoftext|>Similarity: {sim}, Aesthetic: {aes}, Query: {cond}",
        }
        if model_name not in PROMPT_TEMPLATES:
            raise ValueError(f"Invalid model name: {model_name}. Valid: {list(PROMPT_TEMPLATES)}")
        prompt = PROMPT_TEMPLATES[model_name]

        def apply_prompt_template(sample):
            sim = args.sim_level
            aes = args.aes_level
            iqa = args.iqa_level
            con_str = sample["text"]
            
            return {"prompt": prompt.format(aes=aes, iqa=iqa, sim=sim, cond=con_str)}

        dataset = dataset.map(apply_prompt_template, remove_columns=["text"]) 
        queries = query_completion(model_name, dataset['prompt'], args, text_dir)

        
        autocompleted_queries = []

        for item in queries:
            query_text = item[0].split('Query: ')[1] 
            autocompleted_queries.append([query_text])

        ## print some query completion results for illustration
        if args.prt_cmpl_qry:
            print(f"-------------------------   {model_name}   --------------------------------")
            for ii in range(len(autocompleted_queries)):
                for jj, query in enumerate(autocompleted_queries[ii]):
                    print(query)

        if args.search_imgs:
            data_path = resolve_path("..", "processed_data", "coco", "data.pt")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Processed data not found: {data_path}")
            loaded_data = torch.load(data_path, weights_only=False)
            
            ## textual squences to vectors
            q_feats = get_text_list_feature(autocompleted_queries, ai_config, weight_path) 
            
            q_feats = torch.tensor(np.array(q_feats)).squeeze()
            q_feats /= q_feats.norm(dim=-1, keepdim=True)
            q_feats = feat_transform(q_feats.numpy())
            
            
            aesthetics, faiss_smi, iqas, img_hash_list = image_retrieve_coco(args.sear_k, index, q_feats, loaded_data, img_ids)

                
                