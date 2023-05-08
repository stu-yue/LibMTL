import torch
import torch.nn as nn
from module import SKUnit


class SingleModal(nn.Module):
    def __init__(self, **kwargs):
        super(SingleModal, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def forward(self, x):
        """ Input  shape : (bs, 2, T, c) | (bs, 2, c, f, t)
            Output shape : (batch, num_cls) """
        x = x.reshape(len(x), -1, self.input_shape[0], self.input_shape[1])
        out = self.extractor(x)
        return out


class AttentionBiGruExtr(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=256, att_context_size=128, num_layers=1):
        super(AttentionBiGruExtr, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.gru_layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.att_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, att_context_size),   # W*Ht+b,       (batch, seq_len, 2*hidden_size)
                nn.Tanh(),                                      # tanh(W*Ht+b), (batch, seq_len, 2*hidden_size)
                nn.Linear(att_context_size, 1),                 # Ut.T*v,       (batch, seq_len, 1)
                nn.Softmax(dim=1)                               #               (batch, seq_len, 1)
            )
        self.fc_layer = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """ Input  shape : (batch, seq_len, input_size)
            Output shape : (batch, output_size) """
        out, _ = self.gru_layer(x)                      # (batch, seq_len, 2*hidden_size)
        att_out = self.att_layer(out).transpose(1, 2)   # (batch, 1, seq_len)
        out = torch.bmm(att_out, out).squeeze(1)        # (batch, 2*hidden_size)
        if self.output_size < 2 * self.hidden_size:
            out = self.fc_layer(out)                    # (batch, output_size)
        return out


class SKExtr(nn.Module):
    def __init__(self, out_dim, in_feats=2, input_shape=[200, 12], nums_block_list=[2, 2, 2, 2], stride_list=[1, 2, 2, 1], pool_method="gap"):
        super(SKExtr, self).__init__()
        self.pool_method=pool_method
        mid_feats = 32
        self.basic_conv = nn.Sequential(
            nn.Conv2d(in_feats, mid_feats, 7, 1, 3, bias=False),
            nn.BatchNorm2d(mid_feats),
            nn.ReLU(inplace=True)
        )
        self.stage_1 = self._make_layer(mid_feats,  mid_feats,  mid_feats,   nums_block=nums_block_list[0], stride=stride_list[0])
        self.stage_2 = self._make_layer(mid_feats, mid_feats*2, mid_feats*2,  nums_block=nums_block_list[1], stride=stride_list[1])
        self.stage_3 = self._make_layer(mid_feats*2, mid_feats*4, mid_feats*4,  nums_block=nums_block_list[2], stride=stride_list[2])
        self.stage_4 = self._make_layer(mid_feats*4, mid_feats*4, mid_feats*4, nums_block=nums_block_list[3], stride=stride_list[3])
        self.fc = nn.Linear(int(input_shape[1] / 1 + .99) * mid_feats*4, out_dim)
        
    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers=[SKUnit(in_feats, mid_feats, out_feats, stride=(stride, 1), pool_method=self.pool_method)]
        for _ in range(1, nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats, pool_method=self.pool_method))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """ Input shape     : (bz, 3, T, 8)
            Output shape    : (bz, T/4, out_feats) """
        fea = self.basic_conv(x)                # (bz, 32,   T,   8)
        # print(fea.shape)
        fea = self.stage_1(fea)                 # (bz, 32,   T,   8)
        # print(fea.shape)
        fea = self.stage_2(fea)                 # (bz, 64,   T/2, 8)
        # print(fea.shape)
        fea = self.stage_3(fea)                 # (bz, 128,  T/8, 8)
        # print(fea.shape)
        fea = self.stage_4(fea)                 # (bz, 128,  T/8, 8)
        fea = fea.transpose(1, 2).flatten(2, 3)  # (bz, T/4, 1024)
        # print(fea.shape)
        fea = self.fc(fea)                      # (bz, T/4, out_feats)
        return fea


class SelKerGru(SingleModal):
    def __init__(
        self,
        input_shape=[200, 12],
        pool_method="gap",
        **kwargs,
    ):
        super(SelKerGru, self).__init__(**kwargs)
        self.input_shape = input_shape
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.extractor = nn.Sequential(                                             # (bz, 2, T, c)
            SKExtr(out_dim=1024, in_feats=2, input_shape=input_shape, nums_block_list=[1,1,1,1], 
                   stride_list=[1,2,4,1], pool_method=pool_method),                 # (bz, T, dim)
            AttentionBiGruExtr(input_size=1024, hidden_size=512, output_size=512),  # (bz, out_dim)
        )