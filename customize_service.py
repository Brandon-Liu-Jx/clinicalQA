# -*- coding: utf-8 -*-
import logging
import os

import torch
import json
import jieba
from transformers import BertTokenizer, MT5ForConditionalGeneration

from model_service.pytorch_model_service import PTServingBaseService


logger = logging.getLogger(__name__)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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


class QAService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTServingBaseService, self).__init__(model_name, model_path)
        dir_path = os.path.dirname(os.path.realpath(model_path))
        t5_pretrained_path = os.path.join(dir_path, 't5-pegasus')
        self.model = MT5ForConditionalGeneration.from_pretrained(t5_pretrained_path)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(DEVICE)
        self.model.eval()
        self.tokenizer = T5PegasusTokenizer.from_pretrained(t5_pretrained_path)

    def _preprocess(self, data):
        data_dict = data.get('json_line')
        for v in data_dict.values():
            infer_dict = json.loads(v.read())
            return infer_dict

    def _inference(self, data):
        self.model.eval()
        input_data = data.get('question')
        input_ids = self.tokenizer.encode(input_data, max_length=512, truncation='only_first')
        feature = {'input_ids': torch.unsqueeze(torch.LongTensor(input_ids), 0),
                   'attention_mask': torch.unsqueeze(torch.LongTensor([1] * len(input_ids)), 0),
                   'question': input_data,
                   }
        content = {k: v.to(DEVICE) for k, v in feature.items() if k != 'question'}
        output = self.model.generate(max_length=512,
                                     eos_token_id=self.tokenizer.sep_token_id,
                                     decoder_start_token_id=self.tokenizer.cls_token_id,
                                     **content,
                                     )
        infer_result = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        data['answer'] = infer_result[0].replace(' ', '')
        return data

    def _postprocess(self, data):
        return data
