import re
import sys
import io, os
import torch
import numpy as np
import logging
import tqdm
import argparse
from prettytable import PrettyTable
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import glob
import time
import fcntl
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def cal_avg_cosine(k, n=100000):
    cos = torch.nn.CosineSimilarity(dim=-1)
    s = torch.tensor(k[:100000]).cuda()
    kk = []
    pbar = tqdm.tqdm(total=n)
    with torch.no_grad():
        for i in range(n):
            kk.append(cos(s[i:i+1], s).mean().item())
            pbar.set_postfix({'cosine': sum(kk)/len(kk)})
            pbar.update(1)
    return sum(kk) /len(kk)

def s_eval(args):
    se, task = args[0], args[1]
    return se.eval(task)

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def get_delta(model, template, tokenizer, device, args):
    model.eval()

    template = template.replace('*mask*', tokenizer.mask_token)\
                    .replace('*sep+*', '')\
                    .replace('*cls*', '').replace('*sent_0*', ' ')
    # strip for roberta tokenizer
    bs_length = len(tokenizer.encode(template.split(' ')[0].replace('_', ' ').strip())) - 2 + 1
    # replace for roberta tokenizer
    batch = tokenizer([template.replace('_', ' ').strip().replace('   ', ' ')], return_tensors='pt')
    batch['position_ids'] = torch.arange(batch['input_ids'].shape[1]).to(device).unsqueeze(0)
    for k in batch:
        batch[k] = batch[k].repeat(256, 1).to(device)
    batch['position_ids'][:, bs_length:] += torch.arange(256).to(device).unsqueeze(-1)
    m_mask = batch['input_ids'] == tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(**batch,  output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        delta = last_hidden[m_mask]
    delta.requires_grad = False
    template_len = batch['input_ids'].shape[1]
    return delta, template_len

def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_only", action='store_true')
    parser.add_argument('--mlm_head_predict', action='store_true')
    parser.add_argument('--remove_continue_word', action='store_true')
    parser.add_argument('--mask_embedding_sentence', action='store_true')
    parser.add_argument('--mask_embedding_sentence_use_org_pooler', action='store_true')
    parser.add_argument('--mask_embedding_sentence_template', type=str, default=None)
    parser.add_argument('--mask_embedding_sentence_delta', action='store_true')
    parser.add_argument('--mask_embedding_sentence_use_pooler', action='store_true')
    parser.add_argument('--mask_embedding_sentence_autoprompt', action='store_true')
    parser.add_argument('--mask_embedding_sentence_org_mlp', action='store_true')
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str,
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str,
            choices=['cls', 'cls_before_pooler', 'avg',  'avg_first_last'],
            default='cls',
            help="Which pooler to use")
    parser.add_argument("--mode", type=str,
            choices=['dev', 'test', 'fasttest'],
            default='test',
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
            choices=['sts', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--calc_anisotropy', action='store_true')
    parser.add_argument('--dump_sentence_to_embed', action='store_true')
    parser.add_argument('--llama_not_8bit', action='store_true')
    parser.add_argument('--llama_4bit', action='store_true')
    parser.add_argument('--llama_generate', action='store_true')
    parser.add_argument('--lora_weight', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--fix_bos_token', action='store_true')
    parser.add_argument('--no_special_token', action='store_true')
    parser.add_argument('--tensor_parallel', action='store_true')
    parser.add_argument('--template_file', type=str, default=None)
    parser.add_argument('--dense_words', type=str, default=None)

    parser.add_argument('--run_id', type=str, default=None)

    args = parser.parse_args()

    # Load transformers' model checkpoint
    if args.mask_embedding_sentence_org_mlp:
        #only for bert-base
        from transformers import BertForMaskedLM, BertConfig
        config = BertConfig.from_pretrained("bert-base-uncased")
        mlp = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config).cls.predictions.transform
        if 'result' in args.model_name_or_path:
            state_dict = torch.load(args.model_name_or_path+'/pytorch_model.bin')
            new_state_dict = {}
            for key, param in state_dict.items():
                # Replace "mlp" to "pooler"
                if 'pooler' in key:
                    key = key.replace("pooler.", "")
                    new_state_dict[key] = param
            mlp.load_state_dict(new_state_dict)

    isllama = False
    if 'llama' in args.model_name_or_path or 'alpaca' in args.model_name_or_path or\
       'mpt' in args.model_name_or_path or 'opt' in args.model_name_or_path:
        if '*llama_generate*' in args.mask_embedding_sentence_template:
            args.llama_generate = True
            args.mask_embedding_sentence_template = args.mask_embedding_sentence_template.replace('*llama_generate*', '')
            print('use llama generate')

        isllama = True
        from transformers import LlamaTokenizer
        from transformers import LlamaForCausalLM
        if args.tensor_parallel:
            import tensor_parallel as tp

            n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
            if args.llama_4bit:
                from transformers.utils.bitsandbytes import replace_with_bnb_linear, get_keys_to_not_convert, set_module_quantized_tensor_to_device
                model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                             low_cpu_mem_usage = True, torch_dtype=torch.float16)
                modules_to_not_convert = get_keys_to_not_convert(model)
                print(modules_to_not_convert)
                model = tp.TensorParallelPreTrainedModel(model,
                                                         device_ids=[i for i in range(n_gpus)])

                state_dict = model.state_dict()
                model = replace_with_bnb_linear(
                    model,
                    modules_to_not_convert=modules_to_not_convert,
                    quantization_config=transformers.utils.quantization_config.BitsAndBytesConfig(
                        load_in_4bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    ),
                )
                model.is_loaded_4bit = True
                model.is_quantized = True
                for key, param in model.named_parameters():
                    if param.device == torch.device("meta"):
                        if 'wrapped_model.module_shards.1' in key:
                            set_module_quantized_tensor_to_device(model, key, 'cuda:1', value=state_dict[key], fp16_statistics=None)
                        else:
                            set_module_quantized_tensor_to_device(model, key, 'cuda:0', value=state_dict[key], fp16_statistics=None)
                            print(key, param.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                             low_cpu_mem_usage = True, torch_dtype=torch.float16)
                model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
            # model.eval()
        elif '30b' in args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map='auto',
                                                     output_hidden_states=True,
                                                     cache_dir='/home/jt1/llama/30b-cache/',
                                                     load_in_8bit=not args.llama_not_8bit,)
        elif '65b' in args.model_name_or_path:
            print('here')
            from transformers import BitsAndBytesConfig
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_in_4bit=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    #bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                ),
                #torch_dtype=torch.bfloat16,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
        elif args.llama_4bit:
            from transformers import BitsAndBytesConfig
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_in_4bit=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    #bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                ),
                #torch_dtype=torch.bfloat16,
                torch_dtype=torch.float16,
                device_map='auto',
            )
        else:
            #model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,
            #                                         device_map='auto',
            #                                         output_hidden_states=True,
            #                                         load_in_8bit=not args.llama_not_8bit,)
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                         device_map='auto',
                                                         output_hidden_states=True,
                                                         trust_remote_code=True,
                                                         load_in_8bit=not args.llama_not_8bit,)
        if args.lora_weight is not None and len(args.lora_weight) > 0:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                args.lora_weight,
                torch_dtype=torch.float16,
                device_map={'': 0},
            )
            if args.llama_4bit:
                from peft.tuners.lora import LoraLayer
                for name, module in model.named_modules():
                    if isinstance(module, LoraLayer):
                        #module = module.to(torch.bfloat16)
                        module = module.to(torch.float16)
                    if 'norm' in name:
                        module = module.to(torch.float32)
                    if 'lm_head' in name or 'embed_tokens' in name:
                        if hasattr(module, 'weight'):
                            #module = module.to(torch.bfloat16)
                            module = module.to(torch.float16)


        if 'mpt' in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        elif 'opt' in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
            if args.fix_bos_token and tokenizer.bos_token_id == 0:
                print('fix bos token id')
                tokenizer.bos_token_id = 1
                tokenizer.eos_token = '</s>'


        #tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        tokenizer.padding_side = "left"  # Allow batched inference

        # input = tokenizer.batch_encode_plus(['this is test', 'this'], return_tensors='pt',  padding=True, max_length=256, truncation=True)
        # for k in input:
        #     input[k] = input[k].cuda()
        # output = model(**input)
        # import pdb;pdb.set_trace()
    else:
        model = AutoModel.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    if args.mask_embedding_sentence_autoprompt:
        state_dict = torch.load(args.model_name_or_path+'/pytorch_model.bin')
        p_mbv = state_dict['p_mbv']
        template = args.mask_embedding_sentence_template
        template = template.replace('*mask*', tokenizer.mask_token)\
                .replace('*sep+*', '')\
                .replace('*cls*', '').replace('*sent_0*', ' ').replace('_', ' ')
        mask_embedding_template = tokenizer.encode(template)
        mask_index = mask_embedding_template.index(tokenizer.mask_token_id)
        index_mbv = mask_embedding_template[1:mask_index] + mask_embedding_template[mask_index+1:-1]
        #mask_embedding_template = [ 101, 2023, 6251, 1997, 1000, 1000, 2965, 103, 1012, 102]
        #index_mbv = mask_embedding_template[1:7] + mask_embedding_template[8:9]

        dict_mbv = index_mbv
        fl_mbv = [i <= 3 for i, k in enumerate(index_mbv)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #device = torch.device("cpu")
    if not isllama:
        model = model.to(device)
    if args.mask_embedding_sentence_org_mlp:
        mlp = mlp.to(device)

    if args.mask_embedding_sentence_delta:
        delta, template_len = get_delta(model, args.mask_embedding_sentence_template, tokenizer, device, args)

    # Set up the tasks
    #args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    #args.tasks = ['MR']
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            #args.tasks = ['STSBenchmark', 'SICKRelatedness']
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 16}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':16}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    if args.remove_continue_word:
        pun_remove_set = {'?', '*', '#', '´', '’', '=', '…', '|', '~', '/', '‚', '¿', '–', '»', '-', '€', '‘', '"', '(', '•', '`', '$', ':', '[', '”', '%', '£', '<', '[UNK]', ';', '“', '@', '_', '{', '^', ',', '.', '!', '™', '&', ']', '>', '\\', "'", ')', '+', '—'}
        if args.model_name_or_path == 'roberta-base':
            remove_set = {'Ġ.', 'Ġa', 'Ġthe', 'Ġin', 'a', 'Ġ, ', 'Ġis', 'Ġto', 'Ġof', 'Ġand', 'Ġon', 'Ġ\'', 's', '.', 'the', 'Ġman', '-', 'Ġwith', 'Ġfor', 'Ġat', 'Ġwoman', 'Ġare', 'Ġ"', 'Ġthat', 'Ġit', 'Ġdog', 'Ġsaid', 'Ġplaying', 'Ġwas', 'Ġas', 'Ġfrom', 'Ġ:', 'Ġyou', 'Ġan', 'i', 'Ġby'}
        else:
            remove_set = {".", "a", "the", "in", ",", "is", "to", "of", "and", "'", "on", "man", "-", "s", "with", "for", "\"", "at", "##s", "woman", "are", "it", "two", "that", "you", "dog", "said", "playing", "i", "an", "as", "was", "from", ":", "by", "white"}

        vocab = tokenizer.get_vocab()

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        input_sentences = [' '.join(s) for s in batch]
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        if args.mask_embedding_sentence and args.mask_embedding_sentence_template is not None:
            # *cls*_This_sentence_of_"*sent_0*"_means*mask*.*sep+*
            template = args.mask_embedding_sentence_template
            if isllama:
                template = template.replace('_', ' ').replace('*sep+*', '')\
                                                     .replace('*cls*', '')
            else:
                template = template.replace('*mask*', tokenizer.mask_token )\
                                .replace('_', ' ').replace('*sep+*', '')\
                                .replace('*cls*', '')

            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                s = s.replace('"', '\'')
                if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
                sentences[i] = template.replace('*sent 0*', s).strip()
        elif args.remove_continue_word:
            for i, s in enumerate(sentences):
                sentences[i] = ' ' if args.model_name_or_path == 'roberta-base' else ''
                es = tokenizer.encode(' ' + s, add_special_tokens=False)
                for iw, w in enumerate(tokenizer.convert_ids_to_tokens(es)):
                    if args.model_name_or_path == 'roberta-base':
                        # roberta base
                        if 'Ġ' not in w or w in remove_set:
                            pass
                        else:
                            if re.search('[a-zA-Z0-9]', w) is not None:
                                sentences[i] += w.replace('Ġ', '').lower() + ' '
                    elif w not in remove_set and w not in pun_remove_set and '##' not in w:
                        # bert base
                        sentences[i] += w.lower() + ' '
                if len(sentences[i]) == 0: sentences[i] = '[PAD]'

        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                # some llama tokenizer like huggyllama add 1 in before
                #add_special_tokens=False,
                add_special_tokens=not args.no_special_token,
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None

        # Get raw embeddings
        with torch.no_grad():
            if isllama:
                #outputs = model(**batch).hidden_states[-1][:, -1, :]

                if args.llama_generate:
                    model_outputs = model.generate(**batch, max_new_tokens=10, return_dict_in_generate=True, output_hidden_states=True)
                    hidden_states = model_outputs.hidden_states
                    sequences = model_outputs.sequences
                    # not include dot(29889)
                    if True:
                        sequences[sequences < 0] = 0
                        if 'Apple.' in input_sentences[0]:
                            lines = ['Apple.'.join(i.split('Apple.')[1:]) for i in tokenizer.batch_decode(sequences, skip_special_tokens=True)]
                        else:
                            lines = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                        os.makedirs('generate', exist_ok=True)
                        import pdb;pdb.set_trace()
                        with open(os.path.join('generate', args.model_name_or_path.split('/')[-1] + '_generated.txt'), 'a') as f:
                            words = [i.split('means in one word: ')[-1].split('.')[0] for i in lines]
                            for word, s in zip(words, input_sentences):
                                sw = '_'.join(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
                                if 'mpt' in args.model_name_or_path:
                                    f.write(sw.replace('_Ġ', ' ') + '\t' + s + '\n')
                                else:
                                    f.write(sw.replace('_', '') + '\t' + s + '\n')
                    mask = sequences[:, -1] != 29889
                    #outputs = mask.unsqueeze(-1) * hidden_states[1][-1][:, -1, :] +  hidden_states[0][-1][:, -1, :]
                    outputs =  hidden_states[0][-1][:, -1, :]
                    import pdb;pdb.set_trace()
                #elif 'mpt' in args.model_name_or_path:
                    #model_outputs = model.generate(**batch, max_new_tokens=3, return_dict_in_generate=True, output_hidden_states=True)
                elif args.dense_words is not None:
                    word_idx, word_embed = args.word_idx, torch.Tensor(args.word_embed).to(batch['input_ids'].device)
                    word_mask = batch['input_ids'] == word_idx
                    inputs_embeds = model.model.embed_tokens(batch['input_ids'])
                    inputs_embeds[word_mask] = word_embed.half()
                    del batch['input_ids']
                    batch['inputs_embeds'] = inputs_embeds
                    hidden_states = model(**batch, output_hidden_states=True, return_dict=True).hidden_states
                    outputs = hidden_states[-1][:, -1, :]
                    if outputs.dtype == torch.bfloat16:
                        # bfloat16 not support for .numpy()
                        outputs = outputs.float()
                else:
                    hidden_states = model(output_hidden_states=True, return_dict=True, **batch).hidden_states
                    #if 'mpt' in args.model_name_or_path:
                    #    #outputs = hidden_states[-1][:, -1, :]
                    #    import pdb;pdb.set_trace()
                    #    outputs = torch.stack(hidden_states[-1], dim=0)[:, -1, :] #  last layer batch x seq X 4096
                    #else:
                    outputs = hidden_states[-1][:, -1, :]
                    if outputs.dtype == torch.bfloat16:
                        # bfloat16 not support for .numpy()
                        outputs = outputs.float()

                return outputs.cpu()
            elif args.embedding_only:
                hidden_states = None
                pooler_output = None
                last_hidden = model.embeddings.word_embeddings(batch['input_ids'])
                position_ids = model.embeddings.position_ids[:, 0 : last_hidden.shape[1]]
                token_type_ids = torch.zeros(batch['input_ids'].shape, dtype=torch.long,
                                                device=model.embeddings.position_ids.device)

                position_embeddings = model.embeddings.position_embeddings(position_ids)
                token_type_embeddings = model.embeddings.token_type_embeddings(token_type_ids)

                if args.remove_continue_word:
                    batch['attention_mask'][batch['input_ids'] == tokenizer.cls_token_id] = 0
                    batch['attention_mask'][batch['input_ids'] == tokenizer.sep_token_id] = 0
            elif args.mask_embedding_sentence_autoprompt:
                input_ids = batch['input_ids']
                inputs_embeds = model.embeddings.word_embeddings(input_ids)
                p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
                b = torch.arange(input_ids.shape[0]).to(input_ids.device)
                for i, k in enumerate(dict_mbv):
                    if fl_mbv[i]:
                        index = ((input_ids == k) * p).max(-1)[1]
                    else:
                        index = ((input_ids == k) * -p).min(-1)[1]
                    inputs_embeds[b, index] = p_mbv[i]
                batch['input_ids'], batch['inputs_embeds'] = None, inputs_embeds
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                batch['input_ids'] = input_ids

                last_hidden = outputs.last_hidden_state
                pooler_output = last_hidden[input_ids == tokenizer.mask_token_id]

                if args.mask_embedding_sentence_org_mlp:
                    pooler_output = mlp(pooler_output)
                if args.mask_embedding_sentence_delta:
                    blen = batch['attention_mask'].sum(-1) - template_len
                    if args.mask_embedding_sentence_org_mlp:
                        pooler_output -= mlp(delta[blen])
                    else:
                        pooler_output -= delta[blen]
                if args.mask_embedding_sentence_use_pooler:
                    pooler_output = model.pooler.dense(pooler_output)
                    pooler_output = model.pooler.activation(pooler_output)

            else:
                outputs = model(**batch, output_hidden_states=True, return_dict=True)

                try:
                    pooler_output = outputs.pooler_output
                except AttributeError:
                    pooler_output = outputs['last_hidden_state'][:, 0, :]
                if args.mask_embedding_sentence:
                    last_hidden = outputs.last_hidden_state
                    pooler_output = last_hidden[batch['input_ids'] == tokenizer.mask_token_id]
                    if args.mask_embedding_sentence_org_mlp:
                        pooler_output = mlp(pooler_output)
                    if args.mask_embedding_sentence_delta:
                        blen = batch['attention_mask'].sum(-1) - template_len
                        if args.mask_embedding_sentence_org_mlp:
                            pooler_output -= mlp(delta[blen])
                        else:
                            pooler_output -= delta[blen]
                    if args.mask_embedding_sentence_use_org_pooler:
                        pooler_output = mlp(pooler_output)
                    if args.mask_embedding_sentence_use_pooler:
                        pooler_output = model.pooler.dense(pooler_output)
                        pooler_output = model.pooler.activation(pooler_output)
                else:
                    last_hidden = outputs.last_hidden_state
                    hidden_states = outputs.hidden_states


        # Apply different pooler
        if args.mask_embedding_sentence:
            return pooler_output.view(batch['input_ids'].shape[0], -1).cpu()
        elif args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            batch['input_ids'][(batch['input_ids'] == 0) | (batch['input_ids'] == 101) | (batch['input_ids'] == 102)] = batch['input_ids'].max()
            index = batch['input_ids'].topk(3, dim=-1, largest=False)[1]
            index2 = torch.arange(batch['input_ids'].shape[0]).to(index.device)
            r = last_hidden[index2, index[:, 0], :]
            for i in range(1, 3):
                r += last_hidden[index2, index[:, i], :]
            return (r/3).cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError

    if args.dump_sentence_to_embed:
        with open('./data/cs_paper_titles_abs.txt') as f:
            lines = f.readlines()
        batch, embeds = [], []
        print('Get Sentence Embeddings....')
        for line in tqdm.tqdm(lines):
            batch.append(line.replace('\n', '').split())
            if len(batch) >= 128:
                embeds.append(batcher(None, batch, 500).detach().numpy())
                batch = []
        embeds.append(batcher(None, batch, 500).detach().numpy())
        embeds = np.concatenate(embeds, axis=0)
        # save embeds
        np.save('./data/cs_paper_titles_abs_embeds.npy', embeds)
        exit(0)

    if args.calc_anisotropy:
        with open('./data/wiki1m_for_simcse.txt') as f:
            lines = f.readlines()[:100000]
        batch, embeds = [], []
        print('Get Sentence Embeddings....')
        for line in tqdm.tqdm(lines):
            batch.append(line.replace('\n', '').lower().split()[:32])
            if len(batch) >= 128:
                embeds.append(batcher(None, batch).detach().numpy())
                batch = []
        embeds.append(batcher(None, batch).detach().numpy())
        print('Calculate anisotropy....')
        embeds = np.concatenate(embeds, axis=0)
        cosine = cal_avg_cosine(embeds)
        print('Avg. Cos:', cosine)
        exit(0)

    results = {}


    with open(args.template_file) as f:
        all_templates = f.readlines()

    dense_words = None
    if args.dense_words is not None:
        dense_words = np.load(args.dense_words)
        assert dense_words.shape[0] == len(all_templates)

    #if args.template_file is not None:
    for tindex, template in enumerate(all_templates):
        template  = template.replace('\n', '')

        done_template, run_template = set(), set()

        model_name = args.model_name_or_path.split('/')[-1]
        if 't0' in args.template_file:
            model_name += '_t0'
        if args.fix_bos_token:
            model_name += '_fix_bos_token'
        if args.llama_4bit:
            model_name += '_4bit'
        elif args.llama_not_8bit:
            model_name += '_16bit'
        else:
            model_name += '_8bit'
        print(model_name)
        if os.path.exists(os.path.join('dev_results', model_name)):
            with open(os.path.join('dev_results', model_name), 'r') as f:
                done_template = set([i.split(' ')[0] for i in f.readlines()])

        run_template = []
        for run_file in glob.glob(os.path.join('run_dev_' + model_name + '*')):
            with open(run_file, 'r') as f:
                run_template += [i.replace('\n', '') for i in f.readlines()]
        run_template = set(run_template)
        print(run_template)

        if args.dense_words is not None:
            pattern = r'_means_in_one_word:"[^"\n]*"'
            replacement = '_means_in_one_word:"给"'
            args.word_idx = tokenizer.get_vocab()["给"]
            args.word_embed = dense_words[tindex]
            args.mask_embedding_sentence_template = re.sub(pattern, replacement, template)
            template = args.mask_embedding_sentence_template
        else:
            args.mask_embedding_sentence_template = template

        if template in done_template or template in run_template:
            print('skip', template)
            continue

        if args.run_id is None:
            with open('run_dev_' + model_name + '_' + os.environ['CUDA_VISIBLE_DEVICES'].replace(',', ''), 'w') as f:
                f.write(template + '\n')
        else:
            with open('run_dev_' + model_name + '_' + os.environ['CUDA_VISIBLE_DEVICES'].replace(',', '') + '-' + args.run_id, 'w') as f:
                f.write(template + '\n')

        for task in args.tasks:
            se = senteval.engine.SE(params, batcher, prepare)
            result = se.eval(task)
            results[task] = result

        # Print evaluation results
        if args.mode == 'dev':
            print("------ %s ------" % (args.mode))

            task_names = []
            scores = []
            #for task in ['STSBenchmark', 'SICKRelatedness']:
            for task in ['STSBenchmark-dev']:
                task_names.append(task)
                if task in results:
                    scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
                else:
                    scores.append("0.00")
            print_table(task_names, scores)

            if args.mask_embedding_sentence_template is not None:
                if args.checkpoint_path:
                    if os.path.exists(os.path.join(args.checkpoint_path, 'dev_results')):
                        max_scores = 0
                        with open(os.path.join(args.checkpoint_path, 'dev_results'), 'r') as f:
                            for i in f:
                                max_scores = max(max_scores, float(i.split()[1]))
                    else:
                        max_scores = 0
                    if float(scores[-1]) >= max_scores:
                        import shutil
                        if args.lora_weight is not None:
                            shutil.copytree(args.lora_weight, os.path.join(args.checkpoint_path, 'best_model'), dirs_exist_ok=True)
                        else:
                            shutil.copytree(args.model_name_or_path, os.path.join(args.checkpoint_path, 'best_model'), dirs_exist_ok=True)
                    with open(os.path.join(args.checkpoint_path, 'dev_results'), 'a') as f:
                        if args.lora_weight is not None:
                            line = args.mask_embedding_sentence_template + ' ' +str(scores[-1]) + ' ' + args.lora_weight
                        else:
                            line = args.mask_embedding_sentence_template + ' ' +str(scores[-1]) + ' ' + args.model_name_or_path
                        if args.llama_4bit:
                            line += ' 4bit'
                        f.write( line + '\n')
                else:
                    os.makedirs('dev_results', exist_ok=True)
                    line = args.mask_embedding_sentence_template + ' ' +str(scores[-1])
                    lock_and_write_file(os.path.join('dev_results', model_name), line)
                    # with open(os.path.join('dev_results', args.model_name_or_path.split('/')[-1]), 'a') as f:
                    #     line = args.mask_embedding_sentence_template + ' ' +str(scores[-1])
                    #     if args.llama_4bit:
                    #         line += ' 4bit'
                    #     f.write(line + '\n')

            task_names = []
            scores = []
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                task_names.append(task)
                if task in results:
                    scores.append("%.2f" % (results[task]['devacc']))
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)

        elif args.mode == 'test' or args.mode == 'fasttest':
            print("------ %s ------" % (args.mode))

            task_names = []
            scores = []
            for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
                task_names.append(task)
                if task in results:
                    if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                        scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                    else:
                        scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)

            # write results and template to file
            if args.mask_embedding_sentence_template is not None and args.task_set != 'transfer':
                if args.lora_weight is not None:
                    with open(os.path.join('results', args.model_name_or_path.split('/')[-1] + '-lora'), 'a') as f:
                        if args.llama_4bit:
                            f.write(args.mask_embedding_sentence_template + ' ' + args.lora_weight + ' ' +str(scores[-1]) + ' 4bit\n')
                        else:
                            f.write(args.mask_embedding_sentence_template + ' ' + args.lora_weight + ' ' +str(scores[-1]) + '\n')
                elif 'checkpoint' in args.model_name_or_path: # ft
                    with open(os.path.join('results', args.model_name_or_path.split('/')[0]), 'a') as f:
                        f.write(args.mask_embedding_sentence_template + ' ' + args.model_name_or_path.split.split('/')[-1] + ' ' +str(scores[-1]) + '\n')
                else:
                    result_file = os.path.join('results', args.model_name_or_path.split('/')[-1])
                    if args.fix_bos_token:
                        result_file += '-fixbos'
                    with open(result_file, 'a') as f:
                        if args.llama_4bit:
                            f.write(args.mask_embedding_sentence_template + ' ' +str(scores[-1]) + ' 4bit\n')
                        else:
                            f.write(args.mask_embedding_sentence_template + ' ' +str(scores[-1]) + '\n')
                    with open('./sts-org-results', 'a') as f:
                        bits = '16bit' if args.llama_not_8bit else ('4bit' if args.llama_4bit else '8bit')
                        model_name = args.model_name_or_path.split('/')[-1] + f'({bits})'
                        f.write(args.mask_embedding_sentence_template + ' ' + model_name + ' ' + ' '.join([str(s) for s in scores]) + '\n')

            task_names = []
            scores = []
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                task_names.append(task)
                if task in results:
                    scores.append("%.2f" % (results[task]['acc']))
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            if args.task_set == 'transfer':
                with open('./transfer-org-results', 'a') as f:
                    bits = '16bit' if args.llama_not_8bit else ('4bit' if args.llama_4bit else '8bit')
                    model_name = args.model_name_or_path.split('/')[-1] + f'({bits})'
                    f.write(args.mask_embedding_sentence_template + ' ' + model_name + ' ' + ' '.join([str(s) for s in scores]) + '\n')
            print_table(task_names, scores)

if __name__ == "__main__":
    main()
