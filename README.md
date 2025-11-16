# 序列标注作业 - 命名实体识别

基于Bi-LSTM+CRF模型的中文命名实体识别系统，用于识别人名、地名和组织机构名。

## 项目结构

```
.
├── data/                    # 数据目录
│   ├── train_corpus.txt    # 训练语料
│   ├── train_label.txt     # 训练标签
│   ├── test_corpus.txt     # 测试语料
│   └── test_label.txt      # 测试标签
├── data_loader.py          # 数据加载模块
├── model.py                # Bi-LSTM+CRF模型
├── train.py                # 训练脚本
├── evaluate.py             # 评估脚本
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖包
└── README.md               # 说明文档

```

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- NumPy 1.21.0+
- tqdm 4.62.0+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python main.py --mode train
```

或者使用训练脚本直接训练：

```bash
python train.py
```

### 2. 评估模型

```bash
python main.py --mode eval
```

或者使用评估脚本直接评估：

```bash
python evaluate.py
```

### 3. 训练并评估

```bash
python main.py --mode both
```

## 参数说明

### 模型参数

- `--embedding_dim`: 词嵌入维度（默认：100）
- `--hidden_dim`: LSTM隐藏层维度（默认：256）
- `--num_layers`: LSTM层数（默认：1）
- `--dropout`: Dropout比率（默认：0.5）

### 训练参数

- `--batch_size`: 批次大小（默认：32）
- `--num_epochs`: 训练轮数（默认：20）
- `--learning_rate`: 学习率（默认：0.01）
- `--max_len`: 最大序列长度（默认：128）

### 数据路径

- `--train_corpus`: 训练语料路径（默认：data/train_corpus.txt）
- `--train_label`: 训练标签路径（默认：data/train_label.txt）
- `--test_corpus`: 测试语料路径（默认：data/test_corpus.txt）
- `--test_label`: 测试标签路径（默认：data/test_label.txt）

## 标签说明

- `B-PER`: 人名开始
- `I-PER`: 人名中间
- `B-LOC`: 地名开始
- `I-LOC`: 地名中间
- `B-ORG`: 机构名开始
- `I-ORG`: 机构名中间
- `O`: 其他

## 模型结构

本系统采用Bi-LSTM+CRF模型：

1. **词嵌入层**: 将输入字符转换为向量表示
2. **Bi-LSTM层**: 双向LSTM捕获上下文信息
3. **线性层**: 将LSTM输出映射到标签空间
4. **CRF层**: 使用条件随机场进行序列标注，确保标签序列的合理性

## 评估指标

评估脚本会计算以下指标：

1. **包含所有标签（包括O）**: 准确率、召回率、F1值
2. **排除O标签**: 准确率、召回率、F1值
3. **按实体类型**: 人名(PER)、地名(LOC)、机构名(ORG)的详细指标

## 输出文件

训练完成后，会在`checkpoints/`目录下生成：

- `best_model.pth`: 最佳模型权重
- `word2idx.pkl`: 词汇表
- `tag2idx.pkl`: 标签字典
- `idx2tag.pkl`: 标签索引到标签的映射

## 注意事项

1. 训练数据会自动划分20%作为验证集（如果没有提供单独的验证集）
2. 模型使用早停机制，如果验证损失在5个epoch内没有改善，会提前停止训练
3. 训练过程中会自动保存最佳模型（验证损失最低的模型）

## 示例

完整训练和评估流程：

```bash
# 训练模型
python main.py --mode train --num_epochs 20 --batch_size 32

# 评估模型
python main.py --mode eval
```

