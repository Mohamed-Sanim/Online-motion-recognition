from SPDSiamese.ST_TS_HGR_Net import *
from SPDSiamese.spdnet import SPDTangentSpace
import torch
from torch import nn

class ST_TS_SPDC(nn.Module):
    def __init__(self, N, parts, t0=1, NS=15, eps=10**(-4), vect=True, outs = 200):
        super(ST_TS_SPDC, self).__init__()
        self.parts = parts
        self.conv = nn.Conv2d(3, 9, 3, bias=False)
        
        #ST
        self.ga1_st = Gauss_agg1_st(parts)
        self.re_st = ReEig_st()
        self.le_st = LogEig_st()
        self.vm_st = VecMat_st()
        self.ga2_st = Gauss_agg2_st(parts)
        
        #TS
        self.ga1_ts = Gauss_agg1_ts(parts)
        self.re_ts = ReEig_ts(eps)
        self.le_ts = LogEig_ts()
        self.vm_ts = VecMat_ts()
        self.ga2_ts = Gauss_agg2_ts()
        
        #SPDC
        self.spdagg = SPD_Agg(2 * 6 * len(parts), output_size = outs)
        self.lespdc = LogEig_spdc(vect)
        self.fc = nn.Linear((outs * (outs + 1)) // 2, N, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        xx = torch.zeros(x.size(0) * x.size(1), 3, len(self.parts) + 2, len(self.parts[0]) + 2)
        x = torch.cat(tuple([x[:, :, P].unsqueeze(2) for P in self.parts]), 2)
        xx[:, :, 1:-1, 1:-1] = x.squeeze(-1).reshape(x.size(0), x.size(1), x.size(2) * x.size(3),
                                         x.size(4)).transpose(-1, -2).reshape(x.size(0) * x.size(1), 3, len(self.parts), len(self.parts[0]))
        x = self.conv(xx).transpose(1, -1).reshape(x.size(0), x.size(1), len(self.parts) * len(self.parts[0]), 9).unsqueeze(-1)
        
        #ST
        y = self.ga1_st(x)
        y = self.re_st(y)
        y = self.le_st(y[0], y[1], y[2])
        y = self.vm_st(y)
        y = self.ga2_st(y)
        
        #TS
        z = self.ga1_ts(x)
        z = self.re_ts(z)
        z = self.le_ts(z[0], z[1], z[2])
        z = self.vm_ts(z)
        z = self.ga2_ts(z)
        
        #SPDC
        x = torch.cat((y.reshape(y.size(0), 6 * len(self.parts), 56, 56), z.reshape(z.size(0), 6 * len(self.parts), 56, 56)), 1)
        y = self.spdagg(x)
        x = self.lespdc(y)
        x = self.fc(x)
        x = self.sm(x)
        return x, y
    
class Net(nn.Module):
    def __init__(self, N, outs):
        super(Net, self).__init__()
        self.tangent = SPDTangentSpace(outs)
        self.linear = nn.Linear((outs * (outs + 1)) // 2, N, bias=True)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.tangent(x)
        # x = self.dropout(x)
        x = self.linear(x.type(torch.FloatTensor))
        return x
    
    def get_embedding(self,x):
        return self.forward(x)
