
import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import is_main_process

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(*parts):
    return os.path.abspath(os.path.join(BASE_DIR, *parts))

class DataCollatorWithPaddingAndLabels(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)

        # pad labels to same length as input_ids
        if "labels" in features[0]:
            max_len = batch["input_ids"].size(1)
            padded_labels = []

            for f in features:
                labels = f["labels"]
                
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                    
                pad_len = max_len - labels.size(0)
                if pad_len > 0:
                    pad = torch.full(
                        (pad_len,),
                        -100,
                        dtype=labels.dtype,
                    )
                    labels = torch.cat([labels, pad], dim=0)

                padded_labels.append(labels)

            batch["labels"] = torch.stack(padded_labels, dim=0)

        return batch


def main():
    
    parser = argparse.ArgumentParser(description="Dataset-condition query completion")
    
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--warmup', type=int, default=100, help='warmup steps')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    
    parser.add_argument('--logstep', type=int, default=10, help='logging steps')
    parser.add_argument('--savestep', type=int, default=100, help='save steps')
    parser.add_argument('--evalstep', type=int, default=10, help='eval steps')
    parser.add_argument('--eval_strategy', type=str, default='steps', help='evaluation strategy')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler type')
    
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)), help='local rank for distributed training')
    
    parser.add_argument('--project_name', type=str, default='GPT2_COCO', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='gpt2_coco', help='wandb run name')


    args = parser.parse_args()

    set_seed(42)
    
    os.environ["WANDB_PROJECT"] = args.project_name

    ## Environment sanity check (rank 0 only)
    if is_main_process(local_rank=args.local_rank):
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA version:", torch.version.cuda)
    
    
    ## Data Paths
    text_dir = resolve_path('../', 'processed_data', 'coco')
    model_name = "gpt2"
    data_save_path = os.path.join(text_dir, model_name)

    if not os.path.exists(data_save_path):
        raise FileNotFoundError(f"Dataset not found: {data_save_path}")
    
    ## Tokenizer & model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'cls_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'pad_token': '<pad>'})

    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.config.cls_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.eos_token_id 
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    if is_main_process(local_rank=args.local_rank):
        print("Loaded pretrained GPT-2")

    ## Dataset
    tokenized_datasets = load_from_disk(data_save_path)
    # tokenized_datasets = tokenized_datasets.remove_columns(["prompt", "query"])
    tokenized_datasets = tokenized_datasets.remove_columns(
        [c for c in tokenized_datasets["train"].column_names
         if c not in {"input_ids", "attention_mask", "labels"}]
    )
    
    if is_main_process(local_rank=args.local_rank):
        print("Loaded tokenized dataset from disk")
        print(tokenized_datasets)

    ## Data collator
    data_collator = DataCollatorWithPaddingAndLabels(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )
    '''
    # or use:
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        # pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )
    '''
    
    
    ## Training arguments
    output_dir = resolve_path('..', 'outputs', f"{args.project_name}_{args.run_name}")
    os.makedirs(output_dir, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    
    training_args = TrainingArguments(
        output_dir=output_dir,     
        overwrite_output_dir=True,
        
        learning_rate=args.lr,                  
        warmup_steps=args.warmup,  
        lr_scheduler_type=args.lr_scheduler, 
        weight_decay=args.wd,                 
        num_train_epochs=args.epochs,    
        per_device_train_batch_size=args.bs,       
        per_device_eval_batch_size=args.bs,       
        
        eval_strategy=args.eval_strategy,    
        eval_steps=args.evalstep,                 
        
        save_strategy="steps", 
        save_steps=args.savestep,    
        save_total_limit=3,             
        
        logging_steps=args.logstep,                  

        fp16=not use_bf16,
        bf16 = use_bf16,
        
        # report_to="wandb" if is_main_process(args.local_rank) else None, 
        report_to=["wandb"] if is_main_process(local_rank=args.local_rank) else [],
        run_name=args.run_name,
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,                   
        args=training_args,           
        train_dataset=tokenized_datasets["train"],  
        eval_dataset=tokenized_datasets["test"],   
        tokenizer=tokenizer,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=data_collator, 
        ## compute_metrics=compute_metrics,
    )

    trainer.train()
    
    if is_main_process(local_rank=args.local_rank):
        print(f"Training finished. Model saved to:\n{output_dir}")



if __name__ == '__main__':
    
    main()
