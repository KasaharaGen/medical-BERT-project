import torch
import torch.nn as nn
import math

class MyBERTConfig:
    def __init__(self, vocab_size=30000, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, max_position_embeddings=512, intermediate_size=1024, dropout=0.1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.dropout = dropout

class MyEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, token_type_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(pos_ids)
        type_emb = self.token_type_embeddings(token_type_ids)
        x = token_emb + pos_emb + type_emb
        return self.dropout(self.layernorm(x))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:2] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None):
        q = self.transpose_for_scores(self.query(x))
        k = self.transpose_for_scores(self.key(x))
        v = self.transpose_for_scores(self.value(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores += attention_mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, self.all_head_size)
        return self.out(context)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        attn_out = self.dropout(self.attn(x, attention_mask))
        x = self.ln1(x + attn_out)
        ff_out = self.dropout(self.ff(x))
        return self.ln2(x + ff_out)

class MyBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = MyEmbedding(config)
        self.encoder = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        x = self.embeddings(input_ids, token_type_ids)
        for block in self.encoder:
            x = block(x, attention_mask)
        return self.ln(x)

class MyBERTForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = MyBERT(config)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        x = self.bert(input_ids, token_type_ids, attention_mask)
        return self.mlm_head(x)
