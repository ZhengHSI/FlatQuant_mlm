import os
import pickle
import datasets
import random
import transformers
import numpy as np
import torch

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('./datasets/wikitext', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('./datasets/wikitext', split='train')
        # traindata = datasets.load_from_disk('./datasets/wikitext', "wikitext-2-raw-v1", split='train')
        traindata = traindata.filter(lambda x: len(x) > 0)
        traindata = traindata.map(lambda x : {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        valdata = datasets.load_dataset(
        './datasets/allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            './datasets/allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_pile(nsamples, seed, seqlen, tokenizer):
    traindata = datasets.load_dataset("./datasets/pile-val-backup", split="validation")
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')
    # random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

# def get_mme(nsamples, seed, seqlen, processor):
#     dataset = datasets.load_from_disk("/home/workspace/dataset/MME")["test"]
#     dataset = dataset.shuffle(seed=seed)
#     # rng = random.Random(42)
#     samples, num_tokens = [], 0
#     prompts_lists = []
#     input_images_lists = []
#     for index, _data in enumerate(dataset):
#         promt = _data["question"]
#         image = _data["image"]
#         msgs = [{'role': 'user', 'content': "(<image>./</image>)\n"+ promt}]
#         prompts_lists.append(processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
#         input_images_lists.append([image])
#         if index >= nsamples:
#             break
#     inputs = processor(
#         prompts_lists,
#         input_images_lists,
#         max_slice_nums=9,
#         use_image_id=True,
#         return_tensors="pt",
#         max_length=8192
#     )["input_ids"]
#     trainloader = []
#     import torch.nn.functional as F
#     for i in range(inputs.size(0)):  # tensor.size(0) == 33
#         inp = inputs.select(0, i).unsqueeze(0)  # 获取第 i 行并增加一个维度
#         pad_size = seqlen - inp.size(1)
#         # 在右侧填充，左边填充 0，右边填充 pad_size 个值
#         inp = F.pad(inp, (pad_size,0), "constant", 0)
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))
#     return trainloader

def get_mme(nsamples, seed, seqlen, processor):
    import torch.nn.functional as F
    dataset = datasets.load_from_disk("/home/workspace/dataset/MME")["test"]
    dataset = dataset.shuffle(seed=seed)
    rng = random.Random(42)
    samples, num_tokens = [], 0
    inputs_list = []
    for index, _data in enumerate(dataset):
        promt = _data["question"]
        image = _data["image"]
        msgs = [{'role': 'user', 'content': "(<image>./</image>)\n"+ promt}]
        prompts = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if index >= nsamples:
            break

        inputs = processor(
            prompts,
            image,
            return_tensors="pt",
            max_length=8192
        )["input_ids"]
    
        inputs_list.append(inputs)

    trainloader = []
    for i in inputs_list:  # tensor.size(0) == 33
        inp = inputs
        pad_size = seqlen - inp.size(1)
        # 在右侧填充，左边填充 0，右边填充 pad_size 个值
        inp = F.pad(inp, (pad_size,0), "constant", 0)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_ScienceQA(nsamples, seed, seqlen, processor):
    import torch.nn.functional as F
    dataset = datasets.load_from_disk("/home/workspace/dataset/ScienceQA-2")["train"]
    dataset = dataset.shuffle(seed=seed)
    # processor = transformers.AutoProcessor.from_pretrained("/home/workspace/model/MiniCPM-Llama3-V-2_5", trust_remote_code=True)
    rng = random.Random(42)
    samples, num_tokens = [], 0
    inputs_list = []
    for index, _data in enumerate(dataset):
        promt = _data["question"]
        image = _data["image"]
        if image is None:
            nsamples = nsamples + 1
            continue
        msgs = [{'role': 'user', 'content': "(<image>./</image>)\n"+ promt}]
        prompts = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if index >= nsamples:
            break

        inputs = processor(
            prompts,
            image,
            return_tensors="pt",
            max_length=8192
        )["input_ids"]
    
        inputs_list.append(inputs)

    trainloader = []
    for i in inputs_list:  # tensor.size(0) == 33
        inp = inputs
        pad_size = seqlen - inp.size(1)
        # 在右侧填充，左边填充 0，右边填充 pad_size 个值
        inp = F.pad(inp, (pad_size,0), "constant", 0)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

# def get_ScienceQA(nsamples, seed, seqlen, processor):
#     import torch.nn.functional as F
#     dataset = datasets.load_from_disk("/home/workspace/dataset/ScienceQA-2")["train"]
#     dataset = dataset.shuffle(seed=seed)
#     rng = random.Random(42)
#     samples, num_tokens = [], 0
#     prompts_lists = []
#     input_images_lists = []
#     for index, _data in enumerate(dataset):
#         promt = _data["question"]
#         image_file = _data["image"]
#         image = np.array(image_file)
#         if image_file is None:
#             nsamples = nsamples+1
#             break
#         msgs = [{'role': 'user', 'content': "(<image>./</image>)\n"+ promt}]
#         prompts_lists.append(processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
#         input_images_lists.append([image])
#         if index >= nsamples:
#             break
#         inputs = processor(
#             prompts_lists,
#             input_images_lists,
#             return_tensors="pt",
#             max_length=8192
#         )["input_ids"]
#     trainloader = []
    
#     for i in range(inputs.size(0)):  # tensor.size(0) == 33
#         inp = inputs.select(0, i).unsqueeze(0)  # 获取第 i 行并增加一个维度
#         pad_size = seqlen - inp.size(1)
#         # 在右侧填充，左边填充 0，右边填充 pad_size 个值
#         # inp = F.pad(inp, (pad_size,0), "constant", 0)
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))

#     return trainloader

def get_loaders(
    args, name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    cache_dir = os.path.join(args.cache_dir, name)
    os.makedirs(cache_dir, exist_ok=True)
    cached_dataset = os.path.join(cache_dir, "testset.pkl" if eval_mode else f"trainset-{nsamples}-{seed}.pkl")
    # if os.path.exists(cached_dataset):
    if False:
        print(f"Loading cached tokenized dataset at {cached_dataset}...")
        with open(cached_dataset, "rb") as f:
            dataset = pickle.load(f)
    else:
        if hf_token is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
        if 'wikitext2' in name:
            dataset = get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode)
        elif 'ptb' in name:
            dataset = get_ptb_new(nsamples, seed, seqlen, tokenizer, eval_mode)
        elif 'c4' in name:
            dataset = get_c4_new(nsamples, seed, seqlen, tokenizer, eval_mode)
        elif 'pile' in name:
            dataset = get_pile(nsamples, seed, seqlen, tokenizer)
        elif 'mme' in name:
            processor = transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
            dataset = get_mme(nsamples, seed, seqlen, processor)
        elif 'ScienceQA' in name:
            processor = transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
            dataset = get_ScienceQA(nsamples, seed, seqlen, processor)
            # for i in dataset:
            #     print(dataset[0][0].shape)
            #     break
        # with open(cached_dataset, "wb") as f:
        #     print(f"Saving cached tokenized dataset at {cached_dataset}...")
        #     if 'c4' in name and eval_mode:
        #         dataset = dataset.input_ids
        #     pickle.dump(dataset, f)
    if 'c4' in name and eval_mode:
        dataset = dataset.input_ids
        dataset = TokenizerWrapper(dataset)
    return dataset
