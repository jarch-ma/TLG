import torch
import torch.nn as nn
import ot
import numpy as np

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.1):
        '''dim is the length of the input sequences'''

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape           # [Batchsize (B) x num_patch (N) x embed_size (C)]

        # Q matrix [B x N x C] ----> [B x N x h x (C/h)] ----> [B x h x N x S]; S = C/h
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # We use a reduction technique to reduce the computational complex of
        # [B x N x C] ----> [B x N/2 x 2 x h x S] ----> [2 x B x h x N/2 x S]
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # [B x h x N/2 x S], [B x h x N/2 x S]

        # Calculate attention weight [B x h x N x S] x [B x h x S x N/2] = [B x h x N x N/2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Calculate attention [B x h x N x N/2] x [B x h x N/2 x S] = [B x h x N x S]
        # [B x h x N x S] ----> [B x N x h x S] ----> [B x N x (hxS)] = [B x N x C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert num_heads == 1, "currently only implement num_heads==1"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.drop_prob = 0.1

    def forward(self, q, k, v, supp_mask=None, OT=None):

        Q = q.shape[1]  #B,c,5
        B, N, C = k.size()  # [B,S,C,N]
        q = q.view(B, -1, C)  # [B,S*G,C]      #[B,G,C]
        k = k.view(B, -1, C)  # [B,S*N,C]      #[B,S*N,C]
        v = v.view(B, -1, C)  # [B,S*N,C]      #[B,S*N,C]
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, S*G, S*N]

        attn = attn.softmax(dim=-1)  # B,G,N
        attn = self.attn_drop(attn)  # [B,S*G,S*N]
        if OT:
            attn_list= []
            for i in range(attn.shape[0]):
                attn_i = attn[i]  # G,N
                attn_i = torch.where(attn_i <= 0, torch.tensor(1e-6), attn_i) # 确保size 2500
                attn_fg = torch.masked_select(attn_i, attn_i > 0).view(attn_i.shape[0], -1)  # G,K
                
                # 为了flops计算，临时注释掉
                cost = (1-attn_fg).detach().cpu()

                # print(cost)
                r, c = attn_fg.size()
                r = int(r)
                c = int(c)
                # attn_fg_new=ot.sinkhorn(np.ones(r) / r, np.ones(c) / c,np.array(cost),0.5)
                attn_fg_new=ot.sinkhorn(np.ones(r) / r, 
                                        np.ones(c) / c,
                                        cost.detach().cpu().numpy(),
                                        0.5)
                ###### flops改
                '''cost = (1 - attn_fg).detach().cpu().numpy()
                r, c = attn_fg.size()

                a = np.ones(r) / r   # numpy ndarray
                b = np.ones(c) / c   # numpy ndarray
                
                attn_fg_new = ot.sinkhorn(a, b, cost, 0.5)'''
                
                #######                 

                attn_fg_new = torch.Tensor(attn_fg_new).cuda()
                attn_i_new = torch.zeros_like(attn_i)
                attn_i_new[attn_i>0] = attn_fg_new.view(-1)

                attn_list.append(attn_i_new)  #ture
            attn = torch.stack(attn_list,dim=0)
        else:
            pass

        x = (attn @ v)  # [B,S*N,C]
        x = self.proj(x)
        x = self.proj_drop(x)  # [B,S*G,C]
        return x