export WANDB_DISABLED=true

MODEL=$1
TEMPLATE="This_sentence_:_\"*sent_0*\"_means_in_one_word:\""

if [[ $MODEL == opt-125m ]] || [[ $MODEL == opt-350m ]] || [[ $MODEL == opt-1.3b ]] || [[ $MODEL == opt-2.7b ]]; then
    WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 ft_llm.py \
        --base_model   facebook/${MODEL} \
        --data_path 'data/nli_for_simcse.csv' \
        --batch_size 256 \
        --micro_batch_size 128 \
        --num_epochs 1 \
        --learning_rate 5e-4 \
        --cutoff_len 32 \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --output_dir ${MODEL}-lora \
        --mask_embedding_sentence_template $TEMPLATE --use_neg_sentence --use_4bit --save_steps 50
elif [[ $MODEL == opt-6.7b ]] || [[ $MODEL == llama-7b ]]; then
    if [[ $MODEL == llama-7b ]]; then
        BASE_MODEL=decapoda-research/llama-7b-hf
        BASE_MODEL="/home/jt/llama-7b"
    else
        BASE_MODEL=facebook/opt-6.7b
    fi
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1234 ft_llm.py \
        --base_model   $BASE_MODEL \
        --data_path 'data/nli_for_simcse.csv' \
        --batch_size 256 \
        --micro_batch_size 64 \
        --num_epochs 1 \
        --learning_rate 5e-4 \
        --cutoff_len 32 \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --output_dir ${MODEL}-lora  --is_sentemb \
        --mask_embedding_sentence_template $TEMPLATE --use_neg_sentence --use_4bit --save_steps 50
elif [[ $MODEL == opt-13b ]] || [[ $MODEL == llama-13b ]]; then
    if [[ $MODEL == llama-13b ]]; then
        BASE_MODEL=decapoda-research/llama-13b-hf
    else
        BASE_MODEL=facebook/opt-13b
    fi
    WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 ft_llm.py \
        --base_model   $BASE_MODEL \
        --data_path 'data/nli_for_simcse.csv' \
        --batch_size 200 \
        --micro_batch_size 25 \
        --num_epochs 1 \
        --learning_rate 5e-4 \
        --cutoff_len 32 \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --output_dir ${MODEL}-lora  --is_sentemb \
        --mask_embedding_sentence_template $TEMPLATE --use_neg_sentence --use_4bit --save_steps 50
fi
