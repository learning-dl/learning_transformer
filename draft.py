import math
import torch
import torch.nn as nn

class PosEncoder:
    def __init__(self, ndim, seq_max_len):
        pos = torch.arange(1, seq_max_len + 1)
        self.pe_matrix = self.gen_pos_enc(ndim, pos)

    @classmethod
    def gen_pos_enc(cls, ndim, pos):
        """

        :param ndim: 整数
        :param pos: 一维整数类型张量
        :return:
        """
        assert ndim % 2 == 0
        max_idx = ndim // 2
        x = pos.view(-1, 1) / (10000 ** (torch.arange(max_idx) / ndim))
        y_sin = torch.sin(x)
        y_cos = torch.cos(x)
        pe = torch.stack([y_sin, y_cos], dim=-1).flatten(start_dim=1)
        return pe

class SelfAttention(nn.Module):
    def __init__(self, in_dim, qkv_dim):
        super().__init__()
        self.sqrt_qkv_dim = math.sqrt(qkv_dim)
        self.w_q = nn.Linear(in_dim, qkv_dim, bias=False)
        self.w_k = nn.Linear(in_dim, qkv_dim, bias=False)
        self.w_v = nn.Linear(in_dim, qkv_dim, bias=False)

    def forward(self, src):
        seq_len = src.shape[1]
        q = self.w_q(em)
        k = self.w_k(em)
        v = self.w_v(em)
        # 计算自注意力
        zz = []
        for i in range(seq_len):
            atte = torch.softmax((q[:, i: i+1, :] * k).sum(dim=-1) / self.sqrt_qkv_dim, dim=-1)
            z = (atte.unsqueeze(-1) * v).sum(dim=1, keepdim=True)
            zz.append(z)
        zz = torch.cat(zz, dim=1)
        return zz

class MultiHeadAtte(nn.Module):
    def __init__(self, num_head, in_dim, qkv_dim, out_dim):
        super().__init__()
        self.all_atte = [SelfAttention(in_dim, qkv_dim) for _ in range(num_head)]
        self.ffnn = nn.Linear(qkv_dim * num_head, out_dim)

    def forward(self, em):
        z_list = [self_atte_layer(em) for self_atte_layer in self.all_atte]
        z = torch.cat(z_list, dim=-1)
        return self.ffnn(z)

class EncodeBlock(nn.Module):
    def __init__(self, num_head, in_dim, qkv_dim):
        super().__init__()
        self.multi_atte_layer = MultiHeadAtte(num_head, in_dim, qkv_dim, in_dim)
        self.enc_ff = nn.Linear(in_dim, in_dim)

    def forward(self, em):
        z = self.multi_atte_layer(em)
        z = z + em
        return self.enc_ff(z) + z

class DecodeBlock(nn.Module):
    def __init__(self):
        super().__init__()



class Transform(nn.Module):
    def __init__(self, in_dim, qkv_dim, atte_head_num, enc_block_num, pe_matrix_len=100):
        super().__init__()
        self.pos_encoder = PosEncoder(ndim=in_dim, seq_max_len=pe_matrix_len)
        self.encoder = nn.Sequential(*[EncodeBlock(atte_head_num, in_dim, qkv_dim) for _ in range(enc_block_num)])

    def forward(self, src, sos, dst_embedding):
        """

        :param src: batch, seq_len, embedding_ndim
        :return:
        """
        batch, seq_len, dk = src.shape
        # 截取 positional encode 矩阵
        pe = self.pos_encoder.pe_matrix[:src.shape[1], ...].unsqueeze(0).expand(batch, seq_len, -1)
        em = src + pe
        h = self.encoder(em)
        # 解码
        dst_em = dst_embedding(sos)

        return







if __name__ == "__main__":

    pos = torch.arange(1, 10 + 1)
    x = PosEncoder.gen_pos_enc(64, pos)

    trans = Transform(in_dim=128, qkv_dim=32, atte_head_num=6, enc_block_num=6)

    em = torch.randn([2, 10, 128, ])

    pred = trans(em)

