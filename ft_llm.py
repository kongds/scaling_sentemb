import os
import sys
from typing import List

import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import Trainer
import torch.distributed as dist

NIL_DATASET = False

from transformers import LlamaTokenizer, LlamaConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import set_seed

from peft import (
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

@dataclass
class DataCollatorForSeq2SeqForNeg:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        _features = self.tokenizer.pad(
            {'input_ids': [feature['input_ids'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        _features['attention_mask'] = self.tokenizer.pad(
            {'input_ids': [feature['attention_mask'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )['input_ids']
        _features['labels'] = self.tokenizer.pad(
            {'input_ids': [feature['labels'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )['input_ids']
        features = _features


        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class SentembTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.is_nli and self.use_neg_sentence:
            input_ids, labels, neg = inputs["input_ids"], inputs["labels"], inputs['attention_mask']
            labels[labels < 0 ] = 0
            neg[neg < 0] = 0
            # padding tensor length
            mw = max(input_ids.size(1), labels.size(1), neg.size(1))

            pad_size = mw - labels.size(1)
            if pad_size > 0:
                labels = torch.cat([torch.zeros(labels.size(0), pad_size).cuda().long(), labels], dim=1)
            pad_size = mw - input_ids.size(1)
            if pad_size > 0:
                input_ids = torch.cat([torch.zeros(input_ids.size(0), pad_size).cuda().long(), input_ids], dim=1)
            pad_size = mw - neg.size(1)
            if pad_size > 0:
                neg = torch.cat([torch.zeros(neg.size(0), pad_size).cuda().long(), neg], dim=1)

            inputs['input_ids'] = torch.cat([input_ids, labels, neg], dim=0)
            inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
            del inputs['labels']
        elif self.is_nli:
            input_ids, labels = inputs["input_ids"], inputs["labels"]
            labels[labels < 0 ] = 0
            # padding tensor length
            if input_ids.size(1) > labels.size(1):
                pad_size = input_ids.size(1) - labels.size(1)
                labels = torch.cat([torch.zeros(labels.size(0), pad_size).cuda().long(), labels], dim=1)
            else:
                pad_size = labels.size(1) - input_ids.size(1)
                input_ids = torch.cat([torch.zeros(input_ids.size(0), pad_size).cuda().long(), input_ids], dim=1)
            inputs['input_ids'] = torch.cat([input_ids, labels], dim=0)
            inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
            del inputs['labels']
        else:
            inputs['input_ids'] = torch.cat([inputs['input_ids'], inputs['input_ids']], dim=0)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], inputs['attention_mask']], dim=0)
            del inputs['labels']

        if hasattr(self, 'llama_avg') and self.llama_avg:
            hidden_states = model(output_hidden_states=True, return_dict=True, **inputs).hidden_states
            last_layer = hidden_states[-1]
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_layer.shape)
            pooler_output = (last_layer * attention_mask).mean(1)
        else:
            pooler_output = model(output_hidden_states=True, return_dict=True, **inputs).hidden_states[-1][:, -1, :]

        if self.use_neg_sentence:
            batch_size = pooler_output.size(0)//3
            pooler_output = torch.stack([pooler_output[:batch_size],
                                         pooler_output[batch_size:2*batch_size],
                                         pooler_output[2*batch_size:]], dim=1)
            z1, z2, z3 = pooler_output[:,0], pooler_output[:,1], pooler_output[:,2]
        else:
            batch_size = pooler_output.size(0)//2
            pooler_output = torch.stack([pooler_output[:batch_size], pooler_output[batch_size:]], dim=1)
            z1, z2 = pooler_output[:,0], pooler_output[:,1]
        loss_fct = nn.CrossEntropyLoss()

        if dist.is_initialized():
            if self.use_neg_sentence:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        if not hasattr(model, "sim"):
            self.sim = Similarity(temp=0.05)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        if self.use_neg_sentence:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(inputs['input_ids'].device)

        if self.use_neg_sentence:
            z3_weight = 0
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(input_ids.device)
            cos_sim = cos_sim + weights
        loss = loss_fct(cos_sim, labels)
        return (loss, pooler_output) if return_outputs else loss

def generate_sentemb_prompt(data_point, tokenizer, cutoff_len, template, prefix='input'):
    sp = f's{prefix}'
    if sp not in data_point:
        input = tokenizer(
            data_point[prefix],
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        input = tokenizer.decode(input['input_ids'])
        data_point[sp] = input
    else:
        input = data_point[sp]

    template = template.replace('_', ' ').replace('*sep+*', '')\
                                         .replace('*cls*', '')
    return template.replace('*sent 0*', input).strip()

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "data/nli_for_simcse.csv",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 256,
        micro_batch_size: int = 64,
        num_epochs: int = 1,
        learning_rate: float = 5e-4,
        cutoff_len: int = 32,
        # lora hyperparams
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve,
        is_sentemb: bool = False,
        mask_embedding_sentence_template: str = None,
        run_name: str = None,
        use_neg_sentence: bool = False,
        load_kbit: int = 4,
        save_steps: int = 100,
        llama_avg: bool = False,
        seed: int = 42,
):
    global NIL_DATASET
    if 'nli_for_simcse' in data_path:
        NIL_DATASET = True

    group_by_length = False
    train_on_inputs = False
    cutoff_len = 32

    assert load_kbit in [4, 8, 16]

    run_name = data_path.split('.')[0]

    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    set_seed(seed)

    config = None

    if load_kbit == 4:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            config=config,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                #bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            #torch_dtype=torch.bfloat16,
            torch_dtype=torch.float32,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_kbit == 8 ,
            torch_dtype=torch.float16 if load_kbit == 16 else torch.float32,
            device_map=device_map,
        )

    if 'opt' in base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

        if tokenizer.bos_token_id == 0:
            # fix bos token id
            tokenizer.bos_token_id = 1
            tokenizer.eos_token = '</s>'

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True, label_prompt=None, neg_prompt=None):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        if label_prompt:
            label_result = tokenizer(
                label_prompt,
                padding=False,
                return_tensors=None,
            )
            result["labels"] = label_result["input_ids"]
            if neg_prompt:
                neg_result = tokenizer(
                    neg_prompt,
                    padding=False,
                    return_tensors=None,
                )
                result["attention_mask"] = neg_result["input_ids"]
        else:
            result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if NIL_DATASET:
            data_point['input'] = data_point['sent0']
            data_point['output'] = data_point['sent1']
            if use_neg_sentence:
                data_point['neg'] = data_point['hard_neg']

        full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                              mask_embedding_sentence_template,
                                              prefix='input')
        if NIL_DATASET:
            pos_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                                      mask_embedding_sentence_template,
                                                      prefix='output')
            if use_neg_sentence:
                neg_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                                          mask_embedding_sentence_template,
                                                          prefix="neg")

        tokenized_full_prompt = tokenize(full_prompt, False,
                                         label_prompt=None if not NIL_DATASET else pos_full_prompt,
                                         neg_prompt=neg_full_prompt if NIL_DATASET and use_neg_sentence else None)
        if not train_on_inputs and not NIL_DATASET:
            user_prompt = generate_sentemb_prompt({**data_point, "output": ""}, tokenizer, cutoff_len,
                                                  mask_embedding_sentence_template,
                                                  prefix='input')
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if load_kbit == 4:
        model = prepare_model_for_kbit_training(model)
        def find_all_linear_names(model):
            cls = bnb.nn.Linear4bit
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names: # needed for 16-bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)
        target_modules = find_all_linear_names(model)
        print(target_modules)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                #module = module.to(torch.bfloat16)
                module = module.to(torch.float32)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    #module = module.to(torch.bfloat16)
                    module = module.to(torch.float32)
    else:
        if load_kbit == 8:
            model = prepare_model_for_int8_training(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    if 'csv' in data_path:
        data = load_dataset("csv", data_files=data_path)
    else:
        data = load_dataset("json", data_files=data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=25)

    DC_FUN = DataCollatorForSeq2SeqForNeg if NIL_DATASET and use_neg_sentence else transformers.DataCollatorForSeq2Seq
    trainer = SentembTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=100,
            load_best_model_at_end=False,
            #ddp_find_unused_parameters=False if ddp else None,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            run_name=run_name,
            report_to=None,
        ),
        data_collator=DC_FUN(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            #tokenizer, return_tensors="pt", padding=True
        ),
    )
    trainer.tokenizer = tokenizer
    trainer.is_nli = NIL_DATASET
    trainer.use_neg_sentence = use_neg_sentence
    trainer.llama_avg = llama_avg
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
