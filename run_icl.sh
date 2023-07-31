#!/bin/bash

declare -A icl_examples

icl_examples['opt-125m']='This_sentence_:_"A_man_is_smoking."_means_in_one_word:"Smoking".This_sentence_:_"*sent_0*"_means_in_one_word:"'
icl_examples['opt-350m']='This_sentence_:_"A_man_is_playing_on_a_guitar_and_singing."_means_in_one_word:"Music".This_sentence_:_"*sent_0*"_means_in_one_word:"'
icl_examples['opt-1.3b']='This_sentence_:_"relating_to_switzerland_or_its_people."_means_in_one_word:"Swiss".This_sentence_:_"*sent_0*"_means_in_one_word:"'
icl_examples['opt-2.7b']='This_sentence_:_"A_jockey_riding_a_horse."_means_in_one_word:"Equestrian".This_sentence_:_"*sent_0*"_means_in_one_word:"'
icl_examples['opt-6.7b']='This_sentence_:_"The_man_is_riding_a_horse."_means_in_one_word:"Horseback-riding".This_sentence_:_"*sent_0*"_means_in_one_word:"'
icl_examples['opt-13b']='This_sentence_:_"meat_from_a_deer."_means_in_one_word:"Venison".This_sentence_:_"*sent_0*"_means_in_one_word:"'
icl_examples['opt-30b']='This_sentence_:_"The_man_is_riding_a_motorcycle_down_the_road."_means_in_one_word:"Motorcycling".This_sentence_:_"*sent_0*"_means_in_one_word:"'
icl_examples['opt-66b']='This_sentence_:_"of_or_relating_to_tutors_or_tutoring."_means_in_one_word:"Tutorial".This_sentence_:_"*sent_0*"_means_in_one_word:"'

MODEL=${1}

kbit=$2
if [ -z "$kbit" ]; then
    kbit=16
fi

# GPU requirements for different models
# <=13b(4bit): 1 GPU
# >13b(4bit): 2 GPUs
# <=2.7b(16bit): 1 GPU
# <=13b(16bit): 2 GPU
# ==30b(16bit): 4 GPU
# ==66b(16bit): 8 GPU

TEMPLATE=${icl_examples[$MODEL]}
echo $TEMPLATE

args=()

if [[ $MODEL == *33b ]] || [[ $MODEL == *66b ]]; then
    # need at least 8 GPUs for 66b and 4 GPUs fro 33b for 16bit
    if [[ $kbit == 16 ]]; then
        args=(--tensor_parallel)
    fi
else
  echo "Neither 'a' nor 'b' is greater than 10."
fi

python evaluation.py \
    --model_name_or_path facebook/${MODEL} \
    --mode test --mask_embedding_sentence \
    --mask_embedding_sentence_template $TEMPLATE --load_kbit $kbit ${args[@]}
