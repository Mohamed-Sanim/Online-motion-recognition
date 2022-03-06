from ST_TS_HGR_Net import ReEig_st
import torch
A = torch.randn(50,50)
A @= A.T
c = ReEig_st()
B = c(A)
print(B)
