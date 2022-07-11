#dual networkの作成
#id_kindをEmbedderで指定に変更

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os

class Embedder(nn.Module):
    def __init__(self, d_model, id_kind=3):
        super().__init__()
        self.embed = nn.Embedding(id_kind, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_board_len=64):
        super().__init__()
        self.d_model = d_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        pe = torch.zeros(max_board_len, d_model).to(device)

        for pos in range(max_board_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, attention_show_flg=False):

        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if attention_show_flg:
            scores, attention_probs = attention(q, k, v, self.d_k, mask, self.dropout, attention_show_flg)
            concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
            output = self.out(concat)
            return output, attention_probs
        else:
            scores = attention(q, k, v, self.d_k, mask, self.dropout, attention_show_flg)
            concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
            output = self.out(concat)
            return output

def attention(q, k, v, d_k, mask=None, dropout=None, attention_show_flg=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
  
    scores = F.softmax(scores, dim=-1)
  
    if dropout is not None:
        scores = dropout(scores)
    
    if attention_show_flg:
        attention_probs = scores
        output = torch.matmul(scores, v)
        return output, attention_probs
    else:
        output = torch.matmul(scores, v)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
    
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, attention_show_flg=False):
        x2 = self.norm_1(x)
        if attention_show_flg:
            x2, attention_probs = self.attn(x2, x2, x2, mask, attention_show_flg)
            x = x + self.dropout_1(x2)
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ff(x2))
            return x, attention_probs
        else:
            x2 = self.attn(x2, x2, x2, mask, attention_show_flg)
            x = x + self.dropout_1(x2)
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ff(x2))
            return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, d_ff, heads), N)
        self.norm = Norm(d_model)

    def forward(self, x, mask=None, attention_show_flg=False):
        x = self.embed(x)
        x = self.pe(x)
        if attention_show_flg:
            for i in range(self.N):
                x, attention_probs = self.layers[i](x, mask, attention_show_flg)
            return self.norm(x), attention_probs
        else:
            for i in range(self.N):
                x = self.layers[i](x, mask, attention_show_flg)
            return self.norm(x)

class PolicyValue(nn.Module):
    def __init__(self, d_model, pol_output_dim=65, val_output_dim=1, max_board_len=64):
        super().__init__()

        self.max_board_len = max_board_len
        self.pol_output_dim = pol_output_dim
        self.val_output_dim = val_output_dim

        self.policy_layer1 = nn.Linear(d_model, pol_output_dim)
        self.policy_layer2 = nn.Linear(max_board_len, 1)

        self.value_layer1 = nn.Linear(d_model, val_output_dim)
        self.value_layer2 = nn.Linear(max_board_len, 1)
        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, output_softmax=False):
        #本当にclsだけでいいのか
        # x = x[:, 0, :]

        p = F.relu(self.policy_layer1(x))
        p = p.view(-1, self.pol_output_dim, self.max_board_len)
        p = self.policy_layer2(p)
        p = p.view(-1, self.pol_output_dim)

        if output_softmax:
            p = self.softmax(p)

        v = F.relu(self.value_layer1(x))
        v = v.view(-1, self.val_output_dim, self.max_board_len)
        v = self.value_layer2(v)
        # v = self.tanh(v).squeeze(1)
        v = v.view(-1)
        v = self.tanh(v)

        return p, v

class PolicyValueTransfomer(nn.Module):
    def __init__(self, d_model=128, d_ff=512, N=2, heads=8):
        super().__init__()

        self.encoder = Encoder(d_model, d_ff, N, heads)
        self.policyvalue = PolicyValue(d_model)

    def forward(self, x, mask=None, attention_show_flg=False, output_softmax=False):
        if attention_show_flg:
            x, attention_probs = self.encoder(x, mask, attention_show_flg)
            policy, value = self.policyvalue(x, output_softmax)
            return policy, value, attention_probs
        else:
            x = self.encoder(x, mask, attention_show_flg)
            policy, value = self.policyvalue(x, output_softmax)
            return policy, value

# モデルの作成
def dual_network():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PolicyValueTransfomer()
    model = model.to(device)

    #parameter初期化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    os.makedirs('./model/', exist_ok=True)
    torch.save(model.state_dict(), './model/transformer0.pth')

    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    dual_network()
