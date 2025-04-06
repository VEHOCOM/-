import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import jieba

class WeiboDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.emotion_map = {
            'neutral': 0,
            'happy': 1,
            'angry': 2,
            'sad': 3,
            'fear': 4,
            'surprise': 5
        }
        
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)  # 直接加载整个JSON文件
            for item in json_data:    # 遍历JSON数组
                self.data.append({
                    'id': item['id'],
                    'content': item['content'],
                    'label': item['label']
                })
                
                # 构建词表
                words = jieba.lcut(item['content'])
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        words = jieba.lcut(item['content'])
        
        # 处理空文本的情况
        if len(words) == 0:
            words = ['<UNK>']  # 使用 UNK 标记替代空文本
        
        # 将文本转换为索引序列
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # 转换为张量
        text_tensor = torch.LongTensor(indices)
        label_tensor = torch.LongTensor([self.emotion_map[item['label']]])
        
        return text_tensor, label_tensor

    @staticmethod
    def collate_fn(batch):
        texts, labels = zip(*batch)
        
        # 计算每个序列的长度
        lengths = torch.LongTensor([len(text) for text in texts])
        
        # 对文本进行填充
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
        labels = torch.cat(labels)
        
        return texts_padded, labels, lengths