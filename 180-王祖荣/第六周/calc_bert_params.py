from transformers import BertModel
import numpy as np

# 指定模型路径
model_path = "H:\\huggingface\\bert-base-chinese"

# 加载预训练的BERT-Base-Chinese模型
model = BertModel.from_pretrained(model_path)

# 获取模型配置
config = model.config

# 根据模型配置获取参数
vocab_size = config.vocab_size  # 词汇表大小
max_seq_length = config.max_position_embeddings  # 序列的最大长度
hidden_dim = config.hidden_size  # Transformer隐藏单元的维度
layer_count = config.num_hidden_layers  # Transformer层的数量
attention_heads = config.num_attention_heads  # 注意力机制头的数量
intermediate_dim = config.intermediate_size  # 前馈网络的中间层维度
print(f"词汇表大小: {vocab_size}")
print(f"序列的最大长度: {max_seq_length}")
print(f"Transformer隐藏单元的维度: {hidden_dim}")
print(f"Transformer层的数量: {layer_count}")
print(f"注意力机制头的数量: {attention_heads}")
print(f"前馈网络的中间层维度: {intermediate_dim}")


# 计算嵌入层的参数量
# 包括词嵌入、位置嵌入和句子类型嵌入的参数量以及LayerNorm的参数量
embedding_params = (
    vocab_size * hidden_dim
    + max_seq_length * hidden_dim
    + 2 * hidden_dim
    + 2 * hidden_dim  # 假设有2种类型的句子嵌入  # LayerNorm的scale和bias
)

# 计算自注意力层的参数量
# 包括Q, K, V的权重和偏置，以及输出的线性层的权重和偏置
attention_params = (hidden_dim * hidden_dim + hidden_dim) * 3  # Q, K, V的权重和偏置

# 计算自注意力输出的参数量
# 包括输出线性层和LayerNorm的参数量
attention_output_params = (
    hidden_dim * hidden_dim
    + hidden_dim
    + 2 * hidden_dim  # 输出线性层的权重和偏置  # LayerNorm的scale和bias
)

# 计算前馈网络的参数量
# 包括两个线性变换的权重和偏置以及LayerNorm的参数量
feed_forward_params = (
    hidden_dim * intermediate_dim
    + intermediate_dim
    + intermediate_dim * hidden_dim  # 第一个线性变换的权重和偏置
    + hidden_dim
    + 2 * hidden_dim  # 第二个线性变换的权重和偏置  # LayerNorm的scale和bias
)


# 计算池化层的参数量
# 通常用于分类任务的池化后全连接层
pooler_params = hidden_dim * hidden_dim + hidden_dim

# 汇总所有参数量
total_params = (
    embedding_params
    + layer_count * (attention_params + attention_output_params + feed_forward_params)
    + pooler_params
)

print(f"手动计算得出的BERT Base Chinese模型的可训练参数总量为: {total_params}")


# 验证
def calculate_trainable_params(model):
    """
    计算模型的可训练参数总量
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# 计算参数总量
total_params = calculate_trainable_params(model)
print(f"使用api获取的BERT-Base-Chinese的可训练参数总量为: {total_params}")
