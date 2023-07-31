#!/bin/bash

# Set variables
DIRECTORY_PATH=$1 # Replace with the path of the directory you want to monitor
CHECK_INTERVAL=5 # Time interval between checks in seconds
MONITOR_PID=$2

# Initialize an empty list of checked directories
declare -A checked_directories
#for entry in "$DIRECTORY_PATH"/*; do
#    if [ -d "$entry" ]; then
#        checked_directories["$entry"]=1
#    fi
#done

eval_results() {
    # Replace this function with the desired action
    MODEL_PATH="decapoda-research/llama-7b-hf"
    if [[ $1 == *"opt-125m"* ]]; then
        MODEL_PATH='facebook/opt-125m'
    elif [[ $1 == *"opt-350m"* ]]; then
        MODEL_PATH='facebook/opt-350m'
    elif [[ $1 == *"opt-1.3b"* ]]; then
        MODEL_PATH='facebook/opt-1.3b'
    elif [[ $1 == *"opt-3b"* ]]; then
        MODEL_PATH='facebook/opt-2.7b'
    elif [[ $1 == *"opt-7b"* ]]; then
        MODEL_PATH='facebook/opt-6.7b'
    elif [[ $1 == *"opt-13b"* ]]; then
        MODEL_PATH='facebook/opt-13b'
    elif [[ $1 == *"llama-13b"* ]]; then
        MODEL_PATH="decapoda-research/llama-13b-hf"
    elif [[ $1 == *"llama-7b"* ]]; then
        MODEL_PATH="decapoda-research/llama-7b-hf"
    fi

    echo $MODEL_PATH

    args=()

    TEMPLATE="This_sentence_:_\"*sent_0*\"_means_in_one_word:\""

    DIR=$(dirname $1)

    if [[ $1 == *"best_model" ]]; then
        python evaluation.py \
            --model_name_or_path   $MODEL_PATH \
            --mode test --mask_embedding_sentence \
            --mask_embedding_sentence_template $TEMPLATE --lora_weight  $1 --load_kbit 16 ${args[@]}
    else
        python evaluation.py \
            --model_name_or_path   $MODEL_PATH \
            --mode dev --mask_embedding_sentence \
            --mask_embedding_sentence_template $TEMPLATE --lora_weight  $1 --load_kbit 4 --checkpoint_path $DIR ${args[@]}
    fi

}


monitor_directory() {
    while true; do
        new_directory_created=false
        for entry in $(ls -tr "$DIRECTORY_PATH"); do
            DIR=$DIRECTORY_PATH/$entry
            echo $DIR
            if [ -d "$DIR" ]; then
                if [[ ! ${checked_directories["$DIR"]} ]]; then
                    checked_directories["$DIR"]=1
                    if [[ $entry != *"best_model"* ]]; then
                        eval_results "$DIR"
                    fi
                fi
            fi
        done

        if [[ -z $MONITOR_PID ]];then
            break
        else
            if kill -0 $MONITOR_PID 2>/dev/null ; then
                sleep $CHECK_INTERVAL
            else
                break
            fi
        fi
    done
}

monitor_directory

eval_results ${DIRECTORY_PATH}/best_model
