import torch
from torch import nn
import numpy as np
import pdb


class SoftGlobalPool(nn.Module):
    def __init__(self):
        super(SoftGlobalPool, self).__init__()
        self.softmax = nn.Softmax2d()
    
    def forward(self, x):
        n, c, h, w = x.shape
        weight = self.softmax(x)                # (bz, c, h, w)
        out = x * weight                        # (bz, c, h, w)
        out = out.sum((-1, -2), keepdim=True)   # (bz, c, 1, 1)
        return out


class MultiSpectralDctPool(nn.Module):
    def __init__(self, freqs=[(0,0)]):
        super(MultiSpectralDctPool, self).__init__()
        self.dct_h = self.dct_w = 7
        self.freqs = freqs
        self.register_buffer("weight", self.get_dct_mask(self.dct_h, self.dct_w, freqs))
        self.fc = nn.Linear(len(freqs), 1)
    
    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        out = x_pooled.unsqueeze(2) * self.weight   # (bz, c, f, dct_h, dct_w)
        out = out.sum((-1, -2))                     # (bz, c, f)
        out = self.fc(out).reshape(n,c,1,1)         # (bz, c, 1, 1)
        return out
    
    def get_1d_dct(self, i, freq, L):
        result = np.cos(np.pi * freq * (i + .5) / L) / np.sqrt(L)
        return result if freq == 0 else result * np.sqrt(2)

    def get_dct_mask(self, height, width, freqs):
        """ 
        width, height, channel  : width, height, channel of input
        fu, fv  : horizontal, vertical indices of selected frequency
        """
        dct_mask = torch.zeros(1, 1, len(freqs), height, width)
        for i, (fu, fv) in enumerate(freqs):
            for h in range(height):
                for w in range(width):
                    dct_mask[:, :, i, h, w] = self.get_1d_dct(h, fu, height) * self.get_1d_dct(w, fv, width)
        return dct_mask
    
    def extra_repr(self):
        return "freqs={}".format(self.freqs)


""" Sequential Input Shape is (N, 3, T, 8) """

class SKConv(nn.Module):
    def __init__(self, features, M=2, G=16, r=16, stride=1 ,L=32, pool_method="gap"):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
            
        if pool_method == "gap":
            self.gp = nn.AdaptiveAvgPool2d((1,1))      # global pool
        elif pool_method == "msd":
            self.gp = MultiSpectralDctPool(freqs=[(0,0), (6,0), (0,6)])
        elif pool_method == "soft":
            self.gp = SoftGlobalPool()
        else:
            raise NotImplementedError
        
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):      # 1x1 conv2d is to fc on channels
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """ tensor shape don't change
        Input shape is (bz, c, h, w)
        Output shape is (bz, c, h, w)  
        """
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]                # feats [(bz, c, h, w), (bz, c, h, w)]
        feats = torch.cat(feats, dim=1)                         # feats (bz, 2*c, h, w)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])   # (bz, 2, c, h, w)
        
        feats_U = torch.sum(feats, dim=1)                       # feats_U (bz, c, h, w)
        feats_S = self.gp(feats_U)                              # feats_S (bz, c, 1, 1)
        feats_Z = self.fc(feats_S)                              # feats_Z (bz, z, 1, 1)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]    # att_vecs [(bz, c, 1, 1), (bz, c, 1, 1)]
        attention_vectors = torch.cat(attention_vectors, dim=1) # att_vecs (bz, 2*c, 1, 1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)     # (bz, 2, c, 1, 1)
        attention_vectors = self.softmax(attention_vectors)     # att_vecs (bz, 2, c, 1, 1), weight(softmax) by branch
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)     # feats_V (bz, c, h, w)
        return feats_V


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=16, r=16, stride=1, L=32, pool_method="gap"):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )

        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L, pool_method=pool_method)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
            )


        if in_features == out_features: # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)         # (bz, mid_feats, h, w) | no  change shape
        out = self.conv2_sk(out)    # (bz, mid_feats, h, w) | may change shape
        out = self.conv3(out)       # (bz, out_feats, h, w) | no  change shape
        
        return self.relu(out + self.shortcut(residual))

class SKNet(nn.Module):
    def __init__(self, class_num, nums_block_list = [3, 4, 6, 3], strides_list = [1, 2, 2, 2]):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.maxpool = nn.MaxPool2d(3,2,1)
        
        self.stage_1 = self._make_layer(64, 128, 256, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(256, 256, 512, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=nums_block_list[3], stride=strides_list[3])
     
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, class_num)
        
    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers=[SKUnit(in_feats, mid_feats, out_feats, stride=stride)]
        for _ in range(1,nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)

    def forward(self, x):
        pdb.set_trace()
        fea = self.basic_conv(x)
        fea = self.maxpool(fea)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)
        fea = self.gap(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea

def SKNet26(nums_class=1000):
    return SKNet(nums_class, [2, 2, 2, 2])
def SKNet50(nums_class=1000):
    return SKNet(nums_class, [3, 4, 6, 3])
def SKNet101(nums_class=1000):
    return SKNet(nums_class, [3, 4, 23, 3])

if __name__=='__main__':
    x = torch.rand(8, 3, 24, 24)
    model = SKNet26()
    out = model(x)

    #flops, params = profile(model, (x, ))
    #flops, params = clever_format([flops, params], "%.5f")
    
    #print(flops, params)
    #print('out shape : {}'.format(out.shape))