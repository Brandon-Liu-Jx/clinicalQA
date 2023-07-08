# coding: UTF-8
# @techreport{zhuiyit5pegasus,
#   title={T5 PEGASUS - ZhuiyiAI},
#   author={Jianlin Su},
#   year={2021},
#   url="https://github.com/ZhuiyiTechnology/t5-pegasus",
# }
import argparse
import re
import json

import torch
import jieba
import numpy as np
from torch._six import container_abcs, string_classes, int_classes
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, BertTokenizer
from torch.utils.data import DataLoader

from data_loader import QADataSet


class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def get_args():
    parser = argparse.ArgumentParser(description='The model for disease diagnosis.')
    parser.add_argument('--train_data_path', default='../../data/train_valid.json', type=str)
    parser.add_argument('--num_epochs', default=20, type=int, help='the epoch of train')
    parser.add_argument('--batch_size', default=8, type=int, help='the batch size of dataset')
    parser.add_argument('--lr', default=2e-4, type=float, help='the learning rate of bert')
    parser.add_argument('--t5_pretrained_path', default='./t5-pegasus')
    parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=512, help='max length of outputs')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='the device gpu or cpu')
    return parser.parse_args()


def compute_bleu(labels, preds, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0.25, 0.25, 0.25, 0.25)
    return np.mean([sentence_bleu(references=[label],
                                  hypothesis=pred,
                                  smoothing_function=SmoothingFunction().method1,
                                  weights=weights) for label, pred in zip(labels, preds)])


def default_collate(batch):
    """
    组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return default_collate([default_collate(elem) for elem in batch])
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def sequence_padding(inputs, length=None, padding=0):
    """
    Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def main():
    args = get_args()
    model = MT5ForConditionalGeneration.from_pretrained(args.t5_pretrained_path).to(args.device)
    model.train()
    tokenizer = T5PegasusTokenizer.from_pretrained(args.t5_pretrained_path)
    with open(args.train_data_path, 'r', encoding='utf8') as rf:
        lines = json.load(rf)
        train_data = lines[:int(0.9*len(lines))]
        valid_data = lines[int(0.9*len(lines)):]
    train_dataset = QADataSet(train_data, tokenizer, mode='train')
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=default_collate,
                            )
    valid_dataset = QADataSet(valid_data, tokenizer, mode='valid')
    valid_iter = DataLoader(dataset=valid_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=default_collate,
                            )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = 0
    for epoch in range(args.num_epochs):
        total_iter = 0
        total_loss = 0.
        for i, cur in enumerate(tqdm(train_iter)):
            model.zero_grad()
            cur = {k: v.to(args.device) for k, v in cur.items()}
            prob = model(**cur)[0]
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            loss.backward()
            optimizer.step()
            total_iter += 1
            total_loss += loss.data
        avg_loss = total_loss / total_iter
        print(f'epoch{epoch}: [Train_Avg_Loss]: {avg_loss}')

        model.eval()
        answers = []
        infer_results = []
        for feature in tqdm(valid_iter):
            answer = feature['answer']
            for result in answer:
                answers.append(jieba.lcut(result.lower()))
            content = {k: v.to(args.device) for k, v in feature.items() if k != 'answer' and k != 'question'}
            gen = model.generate(max_length=args.max_len_generate,
                                 eos_token_id=tokenizer.sep_token_id,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 **content,
                                 )
            infer_result = tokenizer.batch_decode(gen, skip_special_tokens=True)
            infer_result = [jieba.lcut(result.lower()) for result in [item.replace(' ', '') for item in infer_result]]
            infer_results.extend(infer_result)
        scores = compute_bleu(answers, infer_results)
        print(f"epoch{epoch}: Validation scores: {scores}")
        if scores > best:
            best = scores
            torch.save(model.state_dict(), 'baseline_model.pt')
            print('Save Best Model...')


if __name__ == '__main__':
    main()
