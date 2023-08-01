# Scaling Sentence Representation with Large Language Models

## Overview
Large language models (LLMs) have recently garnered significant interest. With in-context learning, LLMs achieve impressive results in various natural language tasks. However, the application of LLMs to sentence embeddings remains an area of ongoing research. In this work, we propose an in-context learning-based method aimed at improving sentence embeddings performance. Our approach involves adapting the previous prompt-based representation method for autore- gressive models, constructing a demonstration set that enables LLMs to perform in-context learning, and scaling up the LLMs to different model sizes. Through extensive experiments, in-context learning enables LLMs to generate high-quality sentence embeddings without any fine-tuning. It helps LLMs achieve performance comparable to current contrastive learning methods. By scaling model size, we find scaling to more than tens of billion parameters harms the performance on semantic textual similarity (STS) tasks. However, the largest model outperforms other counterparts and achieves the new state-of-the-art result on transfer tasks. We also fine-tune LLMs with current contrastive learning approach, and the 2.7B OPT model, incorporating our prompt-based method, surpasses the performance of 4.8B ST5, achieving the new state-of-the-art results on STS tasks.

## Results on STS Tasks with in-context learning (without fine-tuning)

<table align="center">
<thead>
<tr>
<th>Model</th>
<th align="center">STS12</th>
<th align="center">STS13</th>
<th align="center">STS14</th>
<th align="center">STS15</th>
<th align="center">STS16</th>
<th align="center">STSb</th>
<th align="center">SICK-R</th>
<th align="center">Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>OPT 125M</td>
<td align="center">62.22</td>
<td align="center">73.10</td>
<td align="center">61.84</td>
<td align="center">71.09</td>
<td align="center">72.08</td>
<td align="center">67.80</td>
<td align="center">64.10</td>
<td align="center">67.46</td>
</tr>
<tr>
<td>OPT 350M</td>
<td align="center">63.87</td>
<td align="center">73.85</td>
<td align="center">63.41</td>
<td align="center">72.45</td>
<td align="center">73.13</td>
<td align="center">70.84</td>
<td align="center">65.61</td>
<td align="center">69.02</td>
</tr>
<tr>
<td>OPT 1.3B</td>
<td align="center">72.78</td>
<td align="center">83.77</td>
<td align="center">73.61</td>
<td align="center">83.42</td>
<td align="center">80.60</td>
<td align="center">78.80</td>
<td align="center">69.69</td>
<td align="center">77.52</td>
</tr>
<tr>
<td>OPT 2.7B</td>
<td align="center">68.49</td>
<td align="center">84.72</td>
<td align="center">75.15</td>
<td align="center">83.62</td>
<td align="center">81.34</td>
<td align="center">80.94</td>
<td align="center">72.97</td>
<td align="center">78.18</td>
</tr>
<tr>
<td>OPT 6.7B</td>
<td align="center">70.65</td>
<td align="center">84.51</td>
<td align="center">75.01</td>
<td align="center">83.51</td>
<td align="center">82.00</td>
<td align="center">81.12</td>
<td align="center">76.77</td>
<td align="center">79.08</td>
</tr>
<tr>
<td>OPT 13B</td>
<td align="center">71.99</td>
<td align="center">85.22</td>
<td align="center">76.04</td>
<td align="center">82.23</td>
<td align="center">81.38</td>
<td align="center">81.42</td>
<td align="center">75.00</td>
<td align="center">79.04</td>
</tr>
<tr>
<td>OPT 30B</td>
<td align="center">69.99</td>
<td align="center">83.35</td>
<td align="center">74.75</td>
<td align="center">83.14</td>
<td align="center">82.42</td>
<td align="center">81.45</td>
<td align="center">77.46</td>
<td align="center">78.94</td>
</tr>
<tr>
<td>OPT 66B</td>
<td align="center">69.93</td>
<td align="center">83.29</td>
<td align="center">74.88</td>
<td align="center">80.10</td>
<td align="center">81.11</td>
<td align="center">81.76</td>
<td align="center">76.26</td>
<td align="center">78.19</td>
</tr>
</tbody>
</table>

To evaluate the above results, please run the following script, 
```sh
bash run_icl.sh [opt-125m|opt-350m|opt-1.3b|opt-2.7b|opt-6.7b|opt-13b|opt-30b|opt-66b]
```

## Results on STS Tasks with contrastive learning (with fine-tuning)

<table align="center">
<thead>
<tr>
<th align="center">Model</th>
<th align="center">STS12</th>
<th align="center">STS13</th>
<th align="center">STS14</th>
<th align="center">STS15</th>
<th align="center">STS16</th>
<th align="center">STSb</th>
<th align="center">SICK-R</th>
<th align="center">Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><a href="https://huggingface.co/royokong/prompteol-opt-1.3b" rel="nofollow" style="font-size: 0.93em;">royokong/prompteol-opt-1.3b</a></td>
<td align="center">79.01</td>
<td align="center">89.26</td>
<td align="center">84.10</td>
<td align="center">88.30</td>
<td align="center">84.62</td>
<td align="center">87.71</td>
<td align="center">80.52</td>
<td align="center">84.79</td>
</tr>
<tr>
<td align="center"><a href="https://huggingface.co/royokong/prompteol-opt-2.7b" rel="nofollow" style="font-size: 0.93em;">royokong/prompteol-opt-2.7b</a></td>
<td align="center">79.49</td>
<td align="center">89.64</td>
<td align="center">84.80</td>
<td align="center">89.51</td>
<td align="center">85.91</td>
<td align="center">88.33</td>
<td align="center">81.64</td>
<td align="center">85.62</td>
</tr>
<tr>
<td align="center"><a href="https://huggingface.co/royokong/prompteol-opt-6.7b" rel="nofollow" style="font-size: 0.93em;">royokong/prompteol-opt-6.7b</a></td>
<td align="center">80.14</td>
<td align="center">90.02</td>
<td align="center">84.94</td>
<td align="center">89.78</td>
<td align="center">85.84</td>
<td align="center">88.75</td>
<td align="center">81.29</td>
<td align="center">85.82</td>
</tr>
<tr>
<td align="center"><a href="https://huggingface.co/royokong/prompteol-opt-13b" rel="nofollow" style="font-size: 0.93em;">royokong/prompteol-opt-13b</a></td>
<td align="center">80.20</td>
<td align="center">90.24</td>
<td align="center">85.34</td>
<td align="center">89.52</td>
<td align="center">85.90</td>
<td align="center">88.56</td>
<td align="center">82.06</td>
<td align="center">85.97</td>
</tr>
<tr>
<td></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
</tr>
<tr>
<td align="center"><a href="https://huggingface.co/royokong/prompteol-llama-7b" rel="nofollow" style="font-size: 0.93em;">royokong/prompteol-llama-7b</a></td>
<td align="center">79.16</td>
<td align="center">90.22</td>
<td align="center">85.40</td>
<td align="center">88.99</td>
<td align="center">86.25</td>
<td align="center">88.37</td>
<td align="center">81.51</td>
<td align="center">85.70</td>
</tr>
<tr>
<td align="center"><a href="https://huggingface.co/royokong/prompteol-llama-13b" rel="nofollow" style="font-size: 0.93em;">royokong/prompteol-llama-13b</a></td>
<td align="center">78.63</td>
<td align="center">90.03</td>
<td align="center">85.46</td>
<td align="center">89.48</td>
<td align="center">86.18</td>
<td align="center">88.45</td>
<td align="center">82.69</td>
<td align="center">85.85</td>
</tr>
</tbody>
</table>

To evaluate the above results, please run the following script:
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
bash eval_checkpoints.sh opt-2.7b-lora # first evaluate checkpoint on STS-B dev. and evaluate best checkpoint on STS tasks
```

## Acknowledgement
Our Code is based on SimCSE and alpaca-lora
