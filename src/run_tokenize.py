import os
import re
import shutil
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(*parts):
    return os.path.abspath(os.path.join(BASE_DIR, *parts))



def contains_special_characters(text):
    # check if non-ASCII characters exist
    return bool(re.search(r'[^\x00-\x7F]', text))

def check_texts_for_special_characters(texts):
    results = []
    for i, text in enumerate(texts):
        if contains_special_characters(text):
            results.append(f"Text {i}: Contains special characters")
    return results

def clean_text(text):
    # remove non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)  
    # remove redundant space
    text = re.sub(r'\s+', ' ', text)  
    # remove space at the beginning and end of texts
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    return text

def clean_texts(texts):
    return [clean_text(text) for text in texts]




def tokenizing_data_percentile3(tokenizer, data_dict):
    
    dataset = Dataset.from_dict(data_dict)
    
    sim_percentiles = np.percentile(np.array(dataset["similarity"]), [0, 33, 66, 100])
    aes_percentiles = np.percentile(np.array(dataset["aesthetics_score"]), [0, 33, 66, 100])
    iqa_percentiles = np.percentile(np.array(dataset["IQAs"]), [0, 33, 66, 100])
    

    
    def categorize_percentiles(score, percentiles):
        if score <= percentiles[1]:
            return "low"
        elif score <= percentiles[2]:
            return "medium"
        else:
            return "high"
            
    prompt = (
        # f"<|startoftext|>Similarity: {{sim}}, Aesthetic: {{aes}}, Query: "  
        # f"<|startoftext|>DeQA-Score: {{iqa}}, Similarity: {{sim}}, Aesthetic: {{aes}}, Query: "  
        # f"<|startoftext|>Aesthetic: {{aes}}, DeQA-Score: {{iqa}}, Similarity: {{sim}}, Query: "  
        f"<|startoftext|>Similarity: {{sim}}, Aesthetic: {{aes}}, DeQA Quality: {{iqa}}, Query: "  # Condition: {{cond}}, 
    )

    def apply_prompt_template(sample, sim_percentiles, aes_percentiles, iqa_percentiles):    # sim_min, sim_max, aes_min, aes_max, sim_range_step, aes_range_step
        
        sim = categorize_percentiles(sample["similarity"], sim_percentiles)  # categorize_score(sample["similarity"], sim_min, sim_max, sim_range_step)
        aes = categorize_percentiles(sample["aesthetics_score"], aes_percentiles)  #  categorize_score(sample["aesthetics_score"], aes_min, aes_max, aes_range_step)
        iqa = categorize_percentiles(sample["IQAs"], iqa_percentiles) 
        return {
            # "prompt": prompt.format(sim=sim, aes=aes), 
            # "prompt": prompt.format(iqa=iqa, sim=sim, aes=aes), 
            # "prompt": prompt.format(aes=aes, iqa=iqa, sim=sim), 
            "prompt": prompt.format(sim=sim, aes=aes, iqa=iqa), 
            "query": sample["text"],
        }

    dataset = dataset.map(apply_prompt_template, 
                      fn_kwargs={"sim_percentiles": sim_percentiles,
                                 "aes_percentiles": aes_percentiles,
                                 "iqa_percentiles": iqa_percentiles
                                 }) 
    
    
    def tokenize_add_label(sample):

        prompt = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        query = tokenizer.encode(sample["query"], add_special_tokens=False)
        
        text = prompt + query
        
        tokenized_inputs = tokenizer.pad({"input_ids": text}, padding="max_length", max_length=65, return_tensors="pt") # max 58
        
        if tokenized_inputs["input_ids"].shape[0] > 65:  # Check sequence length
            tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"][:65]  # Truncate to max length
            if tokenizer.eos_token_id is not None:  # If EOS token exists
                tokenized_inputs["input_ids"][-1] = tokenizer.eos_token_id  # Set the last token as EOS
            if "attention_mask" in tokenized_inputs:  # Truncate attention_mask if it exists
                tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"][:65]
                
        
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        num_tokens_in_prompt = len(prompt)
        tokenized_inputs["labels"][:num_tokens_in_prompt] = -100
        tokenized_inputs["labels"][tokenized_inputs["labels"] == tokenizer.pad_token_id] = -100
        tokenized_inputs["labels"][tokenized_inputs["labels"] == tokenizer.cls_token_id] = -100
        # tokenized_inputs["labels"][tokenized_inputs["labels"] == tokenizer.eos_token_id] = -100   # calculate the loss on eos token
        
        tokenized_inputs["similarity"] = sample["similarity"] 
        tokenized_inputs["aesthetics_score"] = sample["aesthetics_score"]
        tokenized_inputs["IQAs"] = sample["IQAs"]

        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_add_label, remove_columns=["text"])    # , batched=True, batch_size=1000, num_proc=50
        
    # tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels", "aesthetics_score", "similarity", "prompt", "query"])
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels", "aesthetics_score", "similarity", "IQAs", "prompt", "query"])

    
    return tokenized_datasets




def tokenize_split_save(text_dir, tokenized_data_path, tokenizer):
    
    # data_path = '/home/ubuntu/codes/a1_text/fl/coco_processed_by_aes_IQA_faiss/data.pt'

    data_path = os.path.join(text_dir, "data.pt")
    loaded_data = torch.load(data_path, weights_only=False)


    texts = loaded_data["texts"]
    faiss_sim = loaded_data['faiss_sim']
    aesthetics_score = torch.tensor(loaded_data["aesthetics"])
    IQAs  = torch.tensor(loaded_data["IQAs"])
    
    print(f"data loaded successfully from {data_path}!")
    
    cleaned_texts = [clean_text(text) for text in texts]

    print("Adding eos token at the end for each text...")
    # texts_with_eos = [text + tokenizer.eos_token for text in cleaned_texts]    # add eos_token for each text
    texts_with_eos = [f"{text}<|endoftext|>" for text in cleaned_texts] 
    for ii in range(0, 10):
        print(texts_with_eos[ii])
    
    
    lengths = [len(text) for text in texts_with_eos]
    max_index = lengths.index(max(lengths))
    longest_text = texts_with_eos[max_index]
    longest_text_token = tokenizer.encode(longest_text, return_tensors="pt")
    
    print("Longest text:", longest_text)
    print("Longest text token:", longest_text_token, longest_text_token.shape)
    
    data_dict = {'text': texts_with_eos,
                'similarity': faiss_sim,
                'aesthetics_score': aesthetics_score,
                'IQAs': IQAs
                }
    
    tokenized_datasets = tokenizing_data_percentile3(tokenizer, data_dict)
    
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True, seed=42)     # the order of the data is changed!
    tokenized_datasets = DatasetDict({
        'train': tokenized_datasets['train'],
        'test': tokenized_datasets['test']
    })


    if os.path.exists(tokenized_data_path):     
        shutil.rmtree(tokenized_data_path)      # Removes the directory and all its contents
    tokenized_datasets.save_to_disk(tokenized_data_path)
    print(f"Tokenized data saved to {tokenized_data_path}!")
    
    return tokenized_datasets





if __name__ == '__main__':


    # CUDA_VISIBLE_DEVICES=0 python tokenize.py

    text_dir = resolve_path('../', 'processed_data', 'coco')
    model_name = "gpt2"
    data_save_path = os.path.join(text_dir, model_name)     # ./a1_text/fl/gpt2/

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
            
    tokenizer.add_special_tokens({'cls_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'pad_token': '<pad>'})
    
    model.config.cls_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.eos_token_id 
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
        
    tokenize_split_save(text_dir, data_save_path, tokenizer)

    # tokenized_datasets = tokenized_datasets.remove_columns(["prompt", "query"])
    

                
                
