"""
主程序入口
"""
import argparse
import os
import torch
from train import train_model
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='命名实体识别 - Bi-LSTM+CRF')
    
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'both'],
                       help='运行模式: train(训练), eval(评估), both(训练+评估)')
    
    # 数据路径
    parser.add_argument('--train_corpus', type=str, default='data/train_corpus.txt',
                       help='训练语料路径')
    parser.add_argument('--train_label', type=str, default='data/train_label.txt',
                       help='训练标签路径')
    parser.add_argument('--test_corpus', type=str, default='data/test_corpus.txt',
                       help='测试语料路径')
    parser.add_argument('--test_label', type=str, default='data/test_label.txt',
                       help='测试标签路径')
    
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=100,
                       help='词嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout比率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='学习率')
    parser.add_argument('--max_len', type=int, default=128,
                       help='最大序列长度')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                       help='模型文件路径（评估时使用）')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode == 'train' or args.mode == 'both':
        print("="*60)
        print("开始训练模型")
        print("="*60)
        
        train_model(
            train_corpus_path=args.train_corpus,
            train_label_path=args.train_label,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            max_len=args.max_len,
            save_dir=args.save_dir
        )
    
    if args.mode == 'eval' or args.mode == 'both':
        print("\n" + "="*60)
        print("开始评估模型")
        print("="*60)
        
        if not os.path.exists(args.model_path):
            print(f"错误: 模型文件不存在: {args.model_path}")
            return
        
        evaluate_model(
            model_path=args.model_path,
            test_corpus_path=args.test_corpus,
            test_label_path=args.test_label,
            checkpoint_dir=args.save_dir
        )


if __name__ == '__main__':
    main()

