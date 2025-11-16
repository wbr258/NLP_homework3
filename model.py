"""
Bi-LSTM + CRF 模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


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
        
        # CRF层（使用torchcrf库）
        self.crf = CRF(tag_size, batch_first=True)
        
    def _get_lstm_features(self, sentence, lengths):
        """通过LSTM获取特征"""
        # sentence: [batch_size, seq_len]
        # lengths: [batch_size]
        
        # 词嵌入
        embeds = self.embedding(sentence)  # [batch_size, seq_len, embedding_dim]
        
        # pack_padded_sequence需要lengths在CPU上
        lengths_cpu = lengths.cpu()
        
        # 打包序列（处理变长序列）
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(packed_embeds)
        
        # 解包序列，恢复到原始长度（与sentence相同）
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=sentence.size(1)
        )
        
        # 映射到标签空间
        lstm_feats = self.hidden2tag(lstm_out)  # [batch_size, seq_len, tag_size]
        
        return lstm_feats
    
    
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
        feats = self._get_lstm_features(sentence, lengths)  # [batch_size, seq_len, tag_size]
        
        # pad_packed_sequence返回的长度是batch中最长序列的长度，可能小于sentence的长度
        # 需要确保tags和feats的长度一致
        actual_seq_len = feats.size(1)
        
        if self.training and tags is not None:
            # 训练模式：计算CRF损失
            # 截断tags使其与feats长度一致
            tags = tags[:, :actual_seq_len]
            # torchcrf的CRF层需要mask来标记有效位置
            mask = self._create_mask(lengths, actual_seq_len, feats.device)
            # CRF返回负对数似然，取负号得到损失
            # 如果reduction='mean'不工作，手动取平均
            neg_log_likelihood = self.crf(feats, tags, mask=mask, reduction='mean')
            loss = -neg_log_likelihood
            # 确保loss是标量，明确指定dtype为float32
            if loss.dim() > 0:
                if loss.numel() > 0:
                    loss = loss.mean().float()
                else:
                    loss = torch.tensor(0.0, device=loss.device, dtype=torch.float32)
            return loss
        else:
            # 预测模式：使用CRF解码
            mask = self._create_mask(lengths, actual_seq_len, feats.device)
            best_path = self.crf.decode(feats, mask=mask)
            # 将list转换为tensor，恢复到原始sentence的长度
            original_seq_len = sentence.size(1)
            batch_size = feats.size(0)
            best_path_tensor = torch.zeros(batch_size, original_seq_len, dtype=torch.long).to(feats.device)
            for i, path in enumerate(best_path):
                length = len(path)
                # 只填充实际长度部分，其余保持为0（会被mask忽略）
                if length <= original_seq_len:
                    best_path_tensor[i, :length] = torch.tensor(path, dtype=torch.long).to(feats.device)
                else:
                    best_path_tensor[i, :original_seq_len] = torch.tensor(path[:original_seq_len], dtype=torch.long).to(feats.device)
            return best_path_tensor
    
    def _create_mask(self, lengths, max_len, device):
        """
        创建mask矩阵，标记有效位置
        Args:
            lengths: [batch_size] 每个序列的实际长度
            max_len: 最大序列长度
            device: 设备
        Returns:
            mask: [batch_size, max_len] 布尔tensor，True表示有效位置
        """
        batch_size = lengths.size(0)
        # 使用向量化操作创建mask，更高效
        range_tensor = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = range_tensor < lengths.unsqueeze(1)
        return mask

