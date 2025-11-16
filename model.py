"""
Bi-LSTM + CRF 模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_CRF(nn.Module):
    """Bi-LSTM + CRF 模型用于序列标注"""
    
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=256, 
                 num_layers=1, dropout=0.5):
        """
        Args:
            vocab_size: 词汇表大小
            tag_size: 标签数量
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bi-LSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 线性层，将LSTM输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        
        # CRF层参数：转移矩阵
        # transition[i][j] 表示从标签i转移到标签j的分数
        self.transition = nn.Parameter(torch.randn(tag_size, tag_size))
        
        # 初始化转移矩阵
        # 不允许从非O标签转移到O标签（除非是合理的结束）
        # 不允许从I-X转移到B-Y（X != Y）
        self._init_transition()
        
    def _init_transition(self):
        """初始化CRF转移矩阵"""
        # 将转移矩阵初始化为较小的随机值
        nn.init.uniform_(self.transition, -0.1, 0.1)
        
        # 设置一些约束：不允许从I-X转移到B-Y（X != Y）
        # 这里我们让模型自己学习，不做硬约束
        
    def _get_lstm_features(self, sentence, lengths):
        """通过LSTM获取特征"""
        # sentence: [batch_size, seq_len]
        # lengths: [batch_size]
        
        # 词嵌入
        embeds = self.embedding(sentence)  # [batch_size, seq_len, embedding_dim]
        
        # 打包序列（处理变长序列）
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(packed_embeds)
        
        # 解包序列
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )
        
        # 映射到标签空间
        lstm_feats = self.hidden2tag(lstm_out)  # [batch_size, seq_len, tag_size]
        
        return lstm_feats
    
    def _score_sentence(self, feats, tags, lengths):
        """
        计算给定标签序列的分数
        Args:
            feats: [batch_size, seq_len, tag_size] LSTM输出的特征
            tags: [batch_size, seq_len] 真实标签
            lengths: [batch_size] 每个序列的实际长度
        Returns:
            scores: [batch_size] 每个序列的分数
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        
        scores = torch.zeros(batch_size).to(feats.device)
        
        for i in range(batch_size):
            length = lengths[i].item()
            score = torch.sum(feats[i, range(length), tags[i, :length]])
            
            # 添加转移分数
            for j in range(length - 1):
                score += self.transition[tags[i, j], tags[i, j + 1]]
            
            scores[i] = score
        
        return scores
    
    def _forward_alg(self, feats, lengths):
        """
        前向算法计算所有可能路径的总分数
        Args:
            feats: [batch_size, seq_len, tag_size] LSTM输出的特征
            lengths: [batch_size] 每个序列的实际长度
        Returns:
            alpha: [batch_size] 每个序列的前向分数
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        
        # 初始化alpha：alpha[i][j] 表示在位置i，标签为j的所有路径的总分数
        alpha = torch.full((batch_size, seq_len, tag_size), -1e9).to(feats.device)
        alpha[:, 0, :] = feats[:, 0, :]  # 第一个位置
        
        # 动态规划
        for t in range(1, seq_len):
            for i in range(batch_size):
                if t >= lengths[i]:
                    continue
                # 计算从所有前一个标签转移到当前标签的分数
                for j in range(tag_size):
                    # 前一个位置的所有标签
                    prev_scores = alpha[i, t-1, :] + self.transition[:, j]
                    alpha[i, t, j] = torch.logsumexp(prev_scores, dim=0) + feats[i, t, j]
        
        # 计算每个序列的最终分数
        final_scores = torch.zeros(batch_size).to(feats.device)
        for i in range(batch_size):
            length = lengths[i].item()
            final_scores[i] = torch.logsumexp(alpha[i, length-1, :], dim=0)
        
        return final_scores
    
    def _viterbi_decode(self, feats, lengths):
        """
        Viterbi算法解码，找到最优标签序列
        Args:
            feats: [batch_size, seq_len, tag_size] LSTM输出的特征
            lengths: [batch_size] 每个序列的实际长度
        Returns:
            best_path: [batch_size, seq_len] 最优标签序列
            best_score: [batch_size] 最优路径的分数
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        
        # 初始化
        viterbi = torch.full((batch_size, seq_len, tag_size), -1e9).to(feats.device)
        backpointers = torch.zeros(batch_size, seq_len, tag_size, dtype=torch.long).to(feats.device)
        
        # 第一个位置
        viterbi[:, 0, :] = feats[:, 0, :]
        
        # 动态规划
        for t in range(1, seq_len):
            for i in range(batch_size):
                if t >= lengths[i]:
                    continue
                for j in range(tag_size):
                    # 计算从所有前一个标签转移到当前标签的分数
                    prev_scores = viterbi[i, t-1, :] + self.transition[:, j]
                    best_score, best_prev = torch.max(prev_scores, dim=0)
                    viterbi[i, t, j] = best_score + feats[i, t, j]
                    backpointers[i, t, j] = best_prev
        
        # 回溯找到最优路径
        best_path = torch.zeros(batch_size, seq_len, dtype=torch.long).to(feats.device)
        best_scores = torch.zeros(batch_size).to(feats.device)
        
        for i in range(batch_size):
            length = lengths[i].item()
            # 找到最后一个位置的最优标签
            best_score, best_tag = torch.max(viterbi[i, length-1, :], dim=0)
            best_scores[i] = best_score
            best_path[i, length-1] = best_tag
            
            # 回溯
            for t in range(length-2, -1, -1):
                best_tag = backpointers[i, t+1, best_tag]
                best_path[i, t] = best_tag
        
        return best_path, best_scores
    
    def forward(self, sentence, tags=None, lengths=None):
        """
        前向传播
        Args:
            sentence: [batch_size, seq_len] 输入句子
            tags: [batch_size, seq_len] 真实标签（训练时使用）
            lengths: [batch_size] 每个序列的实际长度
        Returns:
            训练时返回负对数似然损失
            预测时返回最优标签序列
        """
        # 获取LSTM特征
        feats = self._get_lstm_features(sentence, lengths)
        
        if self.training and tags is not None:
            # 训练模式：计算损失
            forward_score = self._forward_alg(feats, lengths)
            gold_score = self._score_sentence(feats, tags, lengths)
            loss = forward_score - gold_score
            return loss.mean()
        else:
            # 预测模式：返回最优路径
            best_path, _ = self._viterbi_decode(feats, lengths)
            return best_path

