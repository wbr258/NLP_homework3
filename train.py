"""
训练脚本
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from data_loader import NERDataset
from model import BiLSTM_CRF


def train_epoch(model, train_loader, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    use_amp = scaler is not None
    
    for batch in tqdm(train_loader, desc="Training"):
        # 使用non_blocking加速数据传输（需要pin_memory配合）
        words = batch['words'].to(device, non_blocking=True)
        tags = batch['tags'].to(device, non_blocking=True)
        # lengths需要确保是tensor格式
        if isinstance(batch['length'], torch.Tensor):
            lengths = batch['length'].to(device, non_blocking=True)
        else:
            # 如果是list或其他格式，转换为tensor
            lengths = torch.tensor(batch['length'], dtype=torch.long).to(device, non_blocking=True)
        
        # 前向传播
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = model(words, tags, lengths)
        else:
            loss = model(words, tags, lengths)
        
        # 反向传播
        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        # 确保loss是标量后再取item()
        if isinstance(loss, torch.Tensor):
            if loss.dim() > 0:
                loss = loss.mean()
            total_loss += loss.item()
        else:
            total_loss += float(loss)
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, val_loader, device):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            words = batch['words'].to(device)
            tags = batch['tags'].to(device)
            # lengths需要确保是tensor格式
            if isinstance(batch['length'], torch.Tensor):
                lengths = batch['length'].to(device)
            else:
                lengths = torch.tensor(batch['length'], dtype=torch.long).to(device)
            
            loss = model(words, tags, lengths)
            # 确保loss是标量后再取item()
            if isinstance(loss, torch.Tensor):
                if loss.dim() > 0:
                    loss = loss.mean()
                total_loss += loss.item()
            else:
                total_loss += float(loss)
            num_batches += 1
    
    return total_loss / num_batches


def train_model(train_corpus_path, train_label_path, 
                val_corpus_path=None, val_label_path=None,
                embedding_dim=100, hidden_dim=256, num_layers=1,
                batch_size=32, num_epochs=20, learning_rate=0.01,
                dropout=0.5, max_len=128, save_dir='./checkpoints',
                use_amp=True):
    """训练模型"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载训练数据
    print("加载训练数据...")
    train_dataset = NERDataset(
        train_corpus_path, 
        train_label_path,
        max_len=max_len
    )
    print("load success...")
    # 保存词汇表和标签表（在划分数据集之前）
    word2idx = train_dataset.word2idx
    tag2idx = train_dataset.tag2idx
    idx2tag = train_dataset.idx2tag
    
    # 如果有验证集，使用验证集；否则从训练集中划分
    if val_corpus_path and val_label_path:
        val_dataset = NERDataset(
            val_corpus_path,
            val_label_path,
            word2idx=word2idx,
            tag2idx=tag2idx,
            max_len=max_len
        )
    else:
        # 从训练集中划分20%作为验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # 优化数据加载：使用多进程和pin_memory加速GPU训练
    # Windows系统使用多进程时需要注意，如果出现问题可以设置num_workers=0
    import sys
    is_windows = sys.platform == 'win32'
    num_workers = 4 if (torch.cuda.is_available() and not is_windows) else 0
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0 and not is_windows)  # Windows上persistent_workers可能有问题
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0 and not is_windows)
    )
    
    # 创建模型
    vocab_size = len(word2idx)
    tag_size = len(tag2idx)
    
    print(f"词汇表大小: {vocab_size}")
    print(f"标签数量: {tag_size}")
    print(f"标签: {list(tag2idx.keys())}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = BiLSTM_CRF(
        vocab_size=vocab_size,
        tag_size=tag_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 混合精度训练的scaler（如果使用GPU且启用AMP）
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None
    if scaler is not None:
        print("启用混合精度训练 (AMP)")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 保存词汇表和标签表
    import pickle
    with open(os.path.join(save_dir, 'word2idx.pkl'), 'wb') as f:
        pickle.dump(word2idx, f)
    with open(os.path.join(save_dir, 'tag2idx.pkl'), 'wb') as f:
        pickle.dump(tag2idx, f)
    with open(os.path.join(save_dir, 'idx2tag.pkl'), 'wb') as f:
        pickle.dump(idx2tag, f)
    
    # 训练循环
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("\n开始训练...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练（使用混合精度训练加速）
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler=scaler)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss = validate(model, val_loader, device)
        print(f"验证损失: {val_loss:.4f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': vocab_size,
                'tag_size': tag_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"保存最佳模型 (验证损失: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"验证损失在{patience}个epoch内没有改善，提前停止训练")
                break
    
    print("\n训练完成！")
    return model, word2idx, tag2idx, idx2tag


if __name__ == '__main__':
    # 训练参数
    train_corpus_path = 'data/train_corpus.txt'
    train_label_path = 'data/train_label.txt'
    
    # 开始训练
    # 如果使用GPU，建议增大batch_size以提高GPU利用率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64 if device.type == 'cuda' else 32
    
    train_model(
        train_corpus_path=train_corpus_path,
        train_label_path=train_label_path,
        embedding_dim=100,
        hidden_dim=256,
        num_layers=1,
        batch_size=batch_size,
        num_epochs=20,
        learning_rate=0.01,
        dropout=0.5,
        max_len=128,
        save_dir='./checkpoints',
        use_amp=True  # 使用混合精度训练加速
    )

