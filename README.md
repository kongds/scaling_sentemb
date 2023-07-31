# Scaling Sentence Representation with Large Language Models

## Overview
Large language models (LLMs) have recently garnered significant interest. With in-context learning, LLMs achieve impressive results in various natural language tasks. However, the application of LLMs to sentence embeddings remains an area of ongoing research. In this work, we propose an in-context learning-based method aimed at improving sentence embeddings performance. Our approach involves adapting the previous prompt-based representation method for autore- gressive models, constructing a demonstration set that enables LLMs to perform in-context learning, and scaling up the LLMs to different model sizes. Through extensive experiments, in-context learning enables LLMs to generate high-quality sentence embeddings without any fine-tuning. It helps LLMs achieve performance comparable to current contrastive learning methods. By scaling model size, we find scaling to more than tens of billion parameters harms the performance on semantic textual similarity (STS) tasks. However, the largest model outperforms other counterparts and achieves the new state-of-the-art result on transfer tasks. We also fine-tune LLMs with current contrastive learning approach, and the 2.7B OPT model, incorporating our prompt-based method, surpasses the performance of 4.8B ST5, achieving the new state-of-the-art results on STS tasks.

## Results on STS Tasks with in-context learing (without fine-tuning)

| Model    | STS12 | STS13 | STS14 | STS15 | STS16 | STSb  | SICK-R | Avg.  |
|----------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|
| OPT 125M | 62.22 | 73.10 | 61.84 | 71.09 | 72.08 | 67.80 | 64.10  | 67.46 |
| OPT 350M | 63.87 | 73.85 | 63.41 | 72.45 | 73.13 | 70.84 | 65.61  | 69.02 |
| OPT 1.3B | 72.78 | 83.77 | 73.61 | 83.42 | 80.60 | 78.80 | 69.69  | 77.52 |
| OPT 2.7B | 68.49 | 84.72 | 75.15 | 83.62 | 81.34 | 80.94 | 72.97  | 78.18 |
| OPT 6.7B | 70.65 | 84.51 | 75.01 | 83.51 | 82.00 | 81.12 | 76.77  | 79.08 |
| OPT 13B  | 71.99 | 85.22 | 76.04 | 82.23 | 81.38 | 81.42 | 75.00  | 79.04 |
| OPT 30B  | 69.99 | 83.35 | 74.75 | 83.14 | 82.42 | 81.45 | 77.46  | 78.94 |
| OPT 66B  | 69.93 | 83.29 | 74.88 | 80.10 | 81.11 | 81.76 | 76.26  | 78.19 |
    
To evaluate the above results, please run the following script, 
```sh
bash run_icl.sh [opt-125m|opt-350m|opt-1.3b|opt-2.7b|opt-6.7b|opt-13b|opt-30b|opt-66b]
```

## Results on STS Tasks with contrastive learing (with fine-tuning)

| Model                                                                               | STS12 | STS13 | STS14 | STS15 | STS16 | STSb  | SICK-R | Avg.  |
|-------------------------------------------------------------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|
| [royokong/prompteol-opt-1.3b](https://huggingface.co/royokong/prompteol-opt-1.3b)   | 79.01 | 89.26 | 84.10 | 88.30 | 84.62 | 87.71 | 80.52  | 84.79 |
| [royokong/prompteol-opt-2.7b](https://huggingface.co/royokong/prompteol-opt-2.7b)   | 79.49 | 89.64 | 84.80 | 89.51 | 85.91 | 88.33 | 81.64  | 85.62 |
| [royokong/prompteol-opt-6.7b](https://huggingface.co/royokong/prompteol-opt-6.7b)   | 80.14 | 90.02 | 84.94 | 89.78 | 85.84 | 88.75 | 81.29  | 85.82 |
| [royokong/prompteol-opt-13b](https://huggingface.co/royokong/prompteol-opt-13b)     | 80.20 | 90.24 | 85.34 | 89.52 | 85.90 | 88.56 | 82.06  | 85.97 |
|                                                                                     |       |       |       |       |       |       |        |       |
| [royokong/prompteol-llama-7b](https://huggingface.co/royokong/prompteol-llama-7b)   | 79.16 | 90.22 | 85.40 | 88.99 | 86.25 | 88.37 | 81.51  | 85.70 |
| [royokong/prompteol-llama-13b](https://huggingface.co/royokong/prompteol-llama-13b) | 78.63 | 90.03 | 85.46 | 89.48 | 86.18 | 88.45 | 82.69  | 85.85 |

To evaluate the above results, please run the following script, 
```sh
MODEL_PATH=facebook/opt-2.7b # or  decapoda-research/llama-x-hf  x model size 7b 13b 
LORA=royokong/prompteol-opt-2.7b # or royokong/prompteol-llama-x x model size 7b 13b
TEMPLATE='This_sentence_:_"*sent_0*"_means_in_one_word:"'
python evaluation.py \
    --model_name_or_path   $MODEL_PATH \
    --mode test --mask_embedding_sentence \
    --mask_embedding_sentence_template $TEMPLATE --lora_weight $LORA --load_kbit 16 
```

## Examples
1. Loading base model
``` python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b")
tokenizer.pad_token_id = 0 
tokenizer.padding_side = "left"
texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]
```

### Use in-context learning to generate embeddings
Directly using in-contex learning get embeddings
``` python
template = 'This_sentence_:_"A_jockey_riding_a_horse."_means_in_one_word:"Equestrian".This_sentence_:_"*sent_0*"_means_in_one_word:"'
inputs = tokenizer([template.replace('*sent_0*', i).replace('_', ' ') for i in texts], padding=True,  return_tensors="pt")
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
```

### Use contrastive learning models to generate embeddings
Using trained LoRA to get embeddings
``` python
from peft import PeftModel
peft_model = PeftModel.from_pretrained(model, "royokong/prompteol-opt-2.7b", torch_dtype=torch.float16)
template = 'This_sentence_:_"*sent_0*"_means_in_one_word:'
inputs = tokenizer([template.replace('*sent_0*', i).replace('_', ' ') for i in texts], padding=True, return_tensors="pt")
with torch.no_grad():
    embeddings = peft_model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
```

## Setup
Install Dependencies

``` sh
pip install -r requirements.txt
```

Download Data

``` sh
cd SentEval/data/downstream/
bash download_dataset.sh
cd -
cd ./data
bash download_nli.sh
cd -
```

## In-context learning
We provide in-context learning examples in `icl_examples.txt`.

To evaluate examples on STS-B development set
``` sh
BASE_MODEL=facebook/opt-2.7b
python evaluation.py \
   --model_name_or_path $BASE_MODEL \
   --mode dev --mask_embedding_sentence \
   --load_kbit 4 --icl_examples_file 274_templates.txt
```

## Contrastive learning
### Train

``` sh
bash train_llm.sh opt-2.7b # can be other models
```

### Test

``` sh
bash eval_checkpoint.sh opt-2.7b-lora # first evaluate checkpoint on STS-B dev. and evaluate best checkpoint on STS tasks
```

## Acknowledgement
Our Code is based on SimCSE and alpaca-lora
