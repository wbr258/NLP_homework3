"""
评估脚本
计算准确率、召回率和F1值
"""
import torch
from torch.utils.data import DataLoader
import pickle
import os
from collections import defaultdict
from data_loader import NERDataset
from model import BiLSTM_CRF


def calculate_metrics(y_true, y_pred, tag2idx, exclude_o=False):
    """
    计算准确率、召回率和F1值
    Args:
        y_true: 真实标签列表（展平的一维列表）
        y_pred: 预测标签列表（展平的一维列表）
        tag2idx: 标签到索引的映射
        exclude_o: 是否排除O标签
    Returns:
        precision, recall, f1: 准确率、召回率、F1值
    """
    # 转换为标签索引
    if isinstance(y_true[0], str):
        y_true = [tag2idx[tag] for tag in y_true]
    if isinstance(y_pred[0], str):
        y_pred = [tag2idx[tag] for tag in y_pred]
    
    # 获取O标签的索引
    o_idx = tag2idx.get('O', -1)
    
    # 统计TP, FP, FN
    tp = defaultdict(int)  # 每个标签的True Positive
    fp = defaultdict(int)  # 每个标签的False Positive
    fn = defaultdict(int)  # 每个标签的False Negative
    
    for true_tag, pred_tag in zip(y_true, y_pred):
        # 如果排除O标签，跳过O标签
        if exclude_o:
            if true_tag == o_idx and pred_tag == o_idx:
                continue
            if true_tag == o_idx:
                continue  # 真实标签是O，预测不是O，不计入FP
            if pred_tag == o_idx:
                continue  # 预测标签是O，真实不是O，不计入FN
        
        if true_tag == pred_tag:
            tp[true_tag] += 1
        else:
            fp[pred_tag] += 1
            fn[true_tag] += 1
    
    # 计算总体指标（宏平均）
    all_tp = sum(tp.values())
    all_fp = sum(fp.values())
    all_fn = sum(fn.values())
    
    if all_tp + all_fp == 0:
        precision = 0.0
    else:
        precision = all_tp / (all_tp + all_fp)
    
    if all_tp + all_fn == 0:
        recall = 0.0
    else:
        recall = all_tp / (all_tp + all_fn)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def evaluate_model(model_path, test_corpus_path, test_label_path, 
                   checkpoint_dir='./checkpoints', device=None):
    """
    评估模型
    Args:
        model_path: 模型文件路径
        test_corpus_path: 测试语料路径
        test_label_path: 测试标签路径
        checkpoint_dir: 检查点目录（包含word2idx和tag2idx）
        device: 计算设备
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 加载词汇表和标签表
    with open(os.path.join(checkpoint_dir, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    with open(os.path.join(checkpoint_dir, 'tag2idx.pkl'), 'rb') as f:
        tag2idx = pickle.load(f)
    with open(os.path.join(checkpoint_dir, 'idx2tag.pkl'), 'rb') as f:
        idx2tag = pickle.load(f)
    
    # 加载测试数据
    print("加载测试数据...")
    test_dataset = NERDataset(
        test_corpus_path,
        test_label_path,
        word2idx=word2idx,
        tag2idx=tag2idx,
        max_len=128
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = BiLSTM_CRF(
        vocab_size=checkpoint['vocab_size'],
        tag_size=checkpoint['tag_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 预测
    print("开始预测...")
    all_true_tags = []  # 所有标签（用于标签级别评估）
    all_pred_tags = []
    all_true_sentences = []  # 按句子分组的标签（用于实体级别评估）
    all_pred_sentences = []
    
    with torch.no_grad():
        for batch in test_loader:
            words = batch['words'].to(device)
            tags = batch['tags'].to(device)
            lengths = batch['length'].to(device)
            
            # 预测
            pred_tags = model(words, lengths=lengths)
            
            # 收集真实标签和预测标签（只取实际长度部分）
            for i in range(words.size(0)):
                length = lengths[i].item()
                true_seq = tags[i, :length].cpu().numpy()
                pred_seq = pred_tags[i, :length].cpu().numpy()
                
                # 转换为标签字符串
                true_tags = [idx2tag[idx] for idx in true_seq]
                pred_tags_str = [idx2tag[idx] for idx in pred_seq]
                
                all_true_tags.extend(true_tags)
                all_pred_tags.extend(pred_tags_str)
                all_true_sentences.append(true_tags)
                all_pred_sentences.append(pred_tags_str)
    
    # 计算指标（包含所有标签）
    print("\n" + "="*50)
    print("评估结果（包含所有标签，包括O）:")
    print("="*50)
    precision_all, recall_all, f1_all = calculate_metrics(
        all_true_tags, all_pred_tags, tag2idx, exclude_o=False
    )
    print(f"准确率 (Precision): {precision_all:.4f}")
    print(f"召回率 (Recall): {recall_all:.4f}")
    print(f"F1值 (F1-Score): {f1_all:.4f}")
    
    # 计算指标（排除O标签）
    print("\n" + "="*50)
    print("评估结果（排除O标签）:")
    print("="*50)
    precision_no_o, recall_no_o, f1_no_o = calculate_metrics(
        all_true_tags, all_pred_tags, tag2idx, exclude_o=True
    )
    print(f"准确率 (Precision): {precision_no_o:.4f}")
    print(f"召回率 (Recall): {recall_no_o:.4f}")
    print(f"F1值 (F1-Score): {f1_no_o:.4f}")
    
    # 按类别计算指标（实体级别，而非标签级别）
    print("\n" + "="*50)
    print("各类别详细指标（实体级别，排除O标签）:")
    print("="*50)
    
    def extract_entities(tags):
        """从标签序列中提取实体，返回(起始位置, 结束位置, 实体类型)的列表"""
        entities = []
        current_entity = None
        current_type = None
        
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                # 开始新实体
                if current_entity is not None:
                    entities.append((current_entity[0], current_entity[1], current_type))
                current_entity = (i, i)
                current_type = tag[2:]
            elif tag.startswith('I-'):
                # 继续当前实体
                if current_entity is not None and tag[2:] == current_type:
                    current_entity = (current_entity[0], i)
                else:
                    # 不匹配，结束当前实体
                    if current_entity is not None:
                        entities.append((current_entity[0], current_entity[1], current_type))
                    current_entity = None
                    current_type = None
            else:
                # O标签，结束当前实体
                if current_entity is not None:
                    entities.append((current_entity[0], current_entity[1], current_type))
                current_entity = None
                current_type = None
        
        # 处理最后一个实体
        if current_entity is not None:
            entities.append((current_entity[0], current_entity[1], current_type))
        
        return entities
    
    # 按实体类型分组统计
    entity_types = ['PER', 'LOC', 'ORG']
    
    for entity_type in entity_types:
        # 提取所有真实实体和预测实体
        all_true_entities = []
        all_pred_entities = []
        
        # 从每个句子中提取实体
        for true_tags, pred_tags in zip(all_true_sentences, all_pred_sentences):
            true_entities = extract_entities(true_tags)
            pred_entities = extract_entities(pred_tags)
            
            # 筛选出当前类型的实体
            all_true_entities.extend([e for e in true_entities if e[2] == entity_type])
            all_pred_entities.extend([e for e in pred_entities if e[2] == entity_type])
        
        # 计算TP, FP, FN（实体必须完全匹配才算正确）
        # 注意：这里简化处理，只比较实体类型和位置范围
        # 实际应该比较实体对应的文本内容
        true_set = set(all_true_entities)
        pred_set = set(all_pred_entities)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{entity_type}:")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  准确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1值: {f1:.4f}")
    
    return {
        'precision_all': precision_all,
        'recall_all': recall_all,
        'f1_all': f1_all,
        'precision_no_o': precision_no_o,
        'recall_no_o': recall_no_o,
        'f1_no_o': f1_no_o
    }


if __name__ == '__main__':
    model_path = './checkpoints/best_model.pth'
    test_corpus_path = 'data/test_corpus.txt'
    test_label_path = 'data/test_label.txt'
    
    evaluate_model(
        model_path=model_path,
        test_corpus_path=test_corpus_path,
        test_label_path=test_label_path
    )

