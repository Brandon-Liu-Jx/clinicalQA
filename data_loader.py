from torch.utils.data import Dataset


class QADataSet(Dataset):

    def __init__(self, input_data, tokenizer, mode='train'):
        self.tokenizer = tokenizer
        self.data = self._create_data(input_data, mode=mode)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _create_data(self, data, mode='train', max_len=512):
        features = []
        for item in data:
            question = item.get('question')
            text_ids = self.tokenizer.encode(question, max_length=max_len, truncation='only_first')
            if mode == 'train':
                answer = item.get('answer')
                summary_ids = self.tokenizer.encode(answer, max_length=max_len, truncation='only_first')
                feature = {'input_ids': text_ids,
                           'decoder_input_ids': summary_ids,
                           'attention_mask': [1] * len(text_ids),
                           'decoder_attention_mask': [1] * len(summary_ids),
                           }
            elif mode == 'valid':
                answer = item.get('answer')
                feature = {'input_ids': text_ids,
                           'attention_mask': [1] * len(text_ids),
                           'answer': answer,
                           'question': question,
                           }
            else:
                feature = {'input_ids': text_ids,
                           'attention_mask': [1] * len(text_ids),
                           }
            features.append(feature)
        return features
