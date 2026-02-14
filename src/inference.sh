#!/bin/bash

: <<'EOF'

EOF


mkdir -p results

levels=(low median high)    #  

models=(model_ADS model_DSA model_SAD model_SA)       # [model_ADS, model_DSA, model_SAD, model_SA]

base_command="CUDA_VISIBLE_DEVICES=0 python ./src/inference.py \
  --cmpl_k 1 \
  --sear_k 1 \
  --min_len 10 \
  --max_len 50 \
  --tmpr 0.7 \
  --top_k 50 \
  --top_p 0.9 \
  --rept_pnal 1.2 \
  --no_rept_ngram 2 \
  --seed 42 \
  --search_imgs"

for model in "${models[@]}"; do
    output_file="./results/finetuned_on_${model}.log"
    for level1 in "${levels[@]}"; do
        for level2 in "${levels[@]}"; do
            for level3 in "${levels[@]}"; do
                full_command="$base_command --model_names ${model} --aes_level ${level1} --sim_level ${level2} --iqa_level ${level3} >> $output_file"
                echo "Now, running: $full_command"
                eval $full_command
                echo "****************************************" >> "$output_file"
            done
        done
    done
done


echo "Test resuts saved to $output_file"

