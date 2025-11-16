"""
数据加载和预处理模块
"""
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np


class NERDataset(Dataset):
    """命名实体识别数据集"""
    
    def __init__(self, corpus_path, label_path, word2idx=None, tag2idx=None, max_len=128):
        """
        Args:
            corpus_path: 语料文件路径
            label_path: 标签文件路径
            word2idx: 词到索引的映射（如果为None，则从数据中构建）
            tag2idx: 标签到索引的映射（如果为None，则从数据中构建）
            max_len: 最大序列长度
        """
        self.max_len = max_len
        
        # 读取数据
        self.sentences, self.labels = self._load_data(corpus_path, label_path)
        
        # 构建词汇表和标签表
        if word2idx is None:
            self.word2idx = self._build_vocab()
        else:
            self.word2idx = word2idx
            
        if tag2idx is None:
            self.tag2idx = self._build_tag_dict()
        else:
            self.tag2idx = tag2idx
            
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        
    def _load_data(self, corpus_path, label_path):
        """加载语料和标签"""
        sentences = []
        labels = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f_corpus, \
             open(label_path, 'r', encoding='utf-8') as f_label:
            
            for line_corpus, line_label in zip(f_corpus, f_label):
                # 去除换行符并按空格分割
                words = line_corpus.strip().split()
                tags = line_label.strip().split()
                
                # 确保长度一致
                if len(words) == len(tags):
                    sentences.append(words)
                    labels.append(tags)
        
        return sentences, labels
    
    def _build_vocab(self):
        """构建词汇表"""
        word_counter = Counter()
        for sentence in self.sentences:
            word_counter.update(sentence)
        
        # 添加特殊标记
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        
        # 添加所有出现过的词
        for word, count in word_counter.items():
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        
        return word2idx
    
    def _build_tag_dict(self):
        """构建标签字典"""
        tag_set = set()
        for tags in self.labels:
            tag_set.update(tags)
        
        # 按照固定顺序构建标签字典
        tag_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
        tag2idx = {}
        for tag in tag_list:
            if tag in tag_set:
                tag2idx[tag] = len(tag2idx)
        
        # 添加其他可能出现的标签
        for tag in sorted(tag_set):
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
        
        return tag2idx
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]
        
        # 转换为索引
        word_indices = [self.word2idx.get(word, self.word2idx['<UNK>']) 
                       for word in sentence]
        tag_indices = [self.tag2idx[tag] for tag in labels]
        
        # 截断或填充到max_len
        if len(word_indices) > self.max_len:
            word_indices = word_indices[:self.max_len]
            tag_indices = tag_indices[:self.max_len]
        else:
            # 填充
            pad_len = self.max_len - len(word_indices)
            word_indices = word_indices + [self.word2idx['<PAD>']] * pad_len
            tag_indices = tag_indices + [self.tag2idx['O']] * pad_len
        
        return {
            'words': torch.LongTensor(word_indices),
            'tags': torch.LongTensor(tag_indices),
            'length': torch.tensor(min(len(self.sentences[idx]), self.max_len), dtype=torch.long)
        }

