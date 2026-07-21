import math
import torch
import torch.nn as nn
from torch.nn.modules.module import T


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

class SDPAttention(nn.Module):
    """
    scaled dot product attention
    """
    def __init__(self, in_dim, qkv_dim):
        super().__init__()
        self.sqrt_qkv_dim = math.sqrt(qkv_dim)
        self.w_q = nn.Linear(in_dim, qkv_dim, bias=False)
        self.w_k = nn.Linear(in_dim, qkv_dim, bias=False)
        self.w_v = nn.Linear(in_dim, qkv_dim, bias=False)

    def forward(self, qx, kvx, right_mask=False):
        """

        :param qx:
        :param kvx:
        :param right_mask: 是否对右侧元素使用 mask
        :return:
        """
        query_len = qx.shape[1]
        q = self.w_q(qx)
        k = self.w_k(kvx)
        v = self.w_v(kvx)
        # 计算自注意力
        zz = []
        for i in range(query_len):
            atte = (q[:, i: i+1, :] * k).sum(dim=-1) / self.sqrt_qkv_dim
            if right_mask:  # 防作弊右侧掩码
                atte[:, i+1:] = -1e9
            atte = torch.softmax(atte, dim=-1)
            z = (atte.unsqueeze(-1) * v).sum(dim=1, keepdim=True)
            zz.append(z)
        zz = torch.cat(zz, dim=1)
        return zz

class MultiHeadAtte(nn.Module):
    def __init__(self, num_head, in_dim, qkv_dim, out_dim):
        super().__init__()
        self.all_atte = nn.ModuleList([SDPAttention(in_dim, qkv_dim) for _ in range(num_head)])
        self.ffn = nn.Linear(qkv_dim * num_head, out_dim)

    def forward(self, em, right_mask=False):
        z_list = [self_atte_layer(em, em, right_mask) for self_atte_layer in self.all_atte]
        z = torch.cat(z_list, dim=-1)
        return self.ffn(z)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_head, in_dim, qkv_dim, out_dim):
        super().__init__()
        self.all_atte = nn.ModuleList([SDPAttention(in_dim, qkv_dim) for _ in range(num_head)])
        self.ffn = nn.Linear(qkv_dim * num_head, out_dim)

    def forward(self, qx, kvx):
        z_list = [self_atte_layer(qx, kvx) for self_atte_layer in self.all_atte]
        z = torch.cat(z_list, dim=-1)
        return self.ffn(z)

class EncodeBlock(nn.Module):
    def __init__(self, num_head, in_dim, qkv_dim):
        super().__init__()
        self.multi_atte_layer = MultiHeadAtte(num_head, in_dim, qkv_dim, in_dim)
        self.enc_ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim)
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, em):
        z = self.multi_atte_layer(em)
        z = self.norm(z + em)
        return self.norm(self.enc_ffn(z) + z)

class DecodeBlock(nn.Module):
    def __init__(self, num_head, in_dim, qkv_dim):
        super().__init__()
        self.multi_atte_layer = MultiHeadAtte(num_head, in_dim, qkv_dim, in_dim)
        self.cross_atte_layer = MultiHeadCrossAttention(num_head, in_dim, qkv_dim, in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim)
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, em, enc_context):
        z = self.multi_atte_layer(em, right_mask=True)
        z = self.norm(z + em)
        y = self.cross_atte_layer(z, enc_context)
        y = self.norm(y + z)
        y_2 = self.ffn(y)
        y_2 = self.norm(y_2 + y)
        return y_2



class Transform(nn.Module):
    def __init__(self, in_dim, qkv_dim, atte_head_num, enc_block_num, dec_block_num, dst_vocab, pe_matrix_len=100):
        super().__init__()
        self.pos_encoder = PosEncoder(ndim=in_dim, seq_max_len=pe_matrix_len)
        self.encoder = nn.Sequential(*[EncodeBlock(atte_head_num, in_dim, qkv_dim) for _ in range(enc_block_num)])
        self.decoder = nn.ModuleList([DecodeBlock(atte_head_num, in_dim, qkv_dim) for _ in range(dec_block_num)])
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, dst_vocab)
        )

    def forward(self, src, sos, dst_embedding, dst_seq=None):
        """

        :param src: batch, seq_len, embedding_ndim
        :return:
        """
        batch, seq_len, dk = src.shape
        # 截取 positional encode 矩阵
        pe = self.pos_encoder.pe_matrix[:src.shape[1], ...].unsqueeze(0).expand(batch, seq_len, -1)
        em = src + pe
        enc_context = self.encoder(em)
        # 解码
        if dst_seq is not None:
            dst_em = dst_embedding(torch.tensor(dst_seq))
            dst_batch = dst_em.shape[0]
            dst_pem = self.pos_encoder.pe_matrix[:dst_em.shape[1], ...].unsqueeze(0).expand(dst_batch, -1, -1)
            dec_context = dst_em + dst_pem
            for decoder in self.decoder:
                dec_context = decoder(dec_context, enc_context)
            return self.ffn(dec_context)



        return





class EnFraTrans(nn.Module):
    def __init__(self, src_vocab_len, dst_vocab_len, em_dim, ):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_len, em_dim)
        self.dst_embedding = nn.Embedding(dst_vocab_len, em_dim)
        self.tf = Transform(em_dim, qkv_dim=64, atte_head_num=6, enc_block_num=6, dec_block_num=6, dst_vocab=dst_vocab_len)

    def forward(self, src, dst=None):
        em = self.src_embedding(torch.tensor(src))
        pred_tf = self.tf(em, 1, self.dst_embedding, dst)

        return pred_tf

    def train_model(self, src_data, dst_data):

        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)

        in_dst = dst_data[:, :-1]
        target = dst_data[:, 1:]
        for _ in range(100):
            pred = self.forward(src_data, in_dst)
            loss = loss_fn(pred.permute(0, 2, 1), target)
            print(f'loss:   {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        return


if __name__ == "__main__":

    src_vocab = 100
    dst_vocab = 100

    src_data = torch.randint(4, 104, [10, 12])
    dst_data = torch.randint(4, 104, [10, 10])
    src_data[:, 0] = 1
    src_data[:, -1] = 2
    dst_data[:, 0] = 1
    dst_data[:, -1] = 2

    translater = EnFraTrans(104, 104, 128)
    translater.train_model(src_data, dst_data)




    print('end')

