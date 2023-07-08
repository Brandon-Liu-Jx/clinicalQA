# coding: UTF-8
import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration

from train import T5PegasusTokenizer, default_collate
from data_loader import QADataSet


def get_args():
    parser = argparse.ArgumentParser(description='The model for disease diagnosis.')
    parser.add_argument('--test_data', default='../data/test.json', type=str)
    parser.add_argument('--batch_size', default=32, type=int, help='the batch size of dataset')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='the device gpu or cpu')
    parser.add_argument('--t5_pretrained_path', default='../t5-pegasus')
    parser.add_argument('--max_len_generate', default=512, help='max length of outputs')
    return parser.parse_args()


def inference():
    args = get_args()
    model = MT5ForConditionalGeneration.from_pretrained(args.t5_pretrained_path).to(args.device)
    model.load_state_dict(torch.load('baseline_model.pt', map_location='cpu'))
    tokenizer = T5PegasusTokenizer.from_pretrained(args.t5_pretrained_path)
    with open(args.test_data, 'r', encoding='utf8') as rf:
        lines = json.load(rf)
    test_dataset = QADataSet(lines, tokenizer, mode='test')
    test_iter = DataLoader(dataset=test_dataset,
                           batch_size=args.batch_size,
                           shuffle=True,
                           collate_fn=default_collate,
                           )
    model.eval()
    model.to(args.device)
    infer_results = []
    with torch.no_grad():
        for feature in test_iter:
            content = {k: v.to(args.device) for k, v in feature.items()}
            gen = model.generate(max_length=args.max_len_generate,
                                 eos_token_id=tokenizer.sep_token_id,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 **content,
                                 )
            infer_result = tokenizer.batch_decode(gen, skip_special_tokens=True)
            infer_results.extend([item.replace(' ', '') for item in infer_result])
    print('return result...')
    print('The first question result :{}'.format(infer_results[0]))


if __name__ == '__main__':
    inference()
