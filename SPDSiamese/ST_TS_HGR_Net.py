import torch
from torch import nn

# ST-TS-HGR-NET Architecture 

## Before ST-TS

def sym(A):
  return 0.5*(A + A.transpose(-1,-2))
def sequence(t):
  d = dict()
  d[0] = range(t)
  d[1] = range(int(t/2))
  d[2] = range(int(t/2),t)
  d[3] = range(int(t/3))
  d[4] = range(int(t/3),2*int(t/3))
  d[5] = range(2*int(t/3),t)
  return d


## ST-GA-NET


### First Gauss Aggregation Layer

class Gauss_agg1_st_function(Function):
    @staticmethod
    def forward(ctx,input,parts, t0):
        t0 = torch.tensor(t0)
        NP, P = len(parts), len(parts[0])
        ctx.save_for_backward(input,t0, torch.tensor(NP), torch.tensor(P))
        batch,nb_frames,joints,coor,col = input.size()
        #ST
        output_st = []
        for s in range(6):
            binf,bsup = min(sequence(nb_frames)[s]),max(sequence(nb_frames)[s])
            x = input[:,binf:bsup + 1].clone().transpose(1,2).reshape(batch,NP, P,bsup-binf + 1,coor,col)
            y = x.clone()
            x[:,:,1:-1] = (x[:,:,1:-1] + x[:,:,:-2] + x[:,:,2:])/3
            mu = x.mean(2)
            cov = torch.zeros(batch,NP,bsup-binf + 1,coor,coor)
            m = mu.unsqueeze(2).expand(x.size())
            xm,x0,xp = y[:,:,:,:-2]-m[:,:,:,1:-1], y-m, y[:,:,:,2:]-m[:,:,:,1:-1]
            cov[:,:,1:-1] = ((xm @ xm.transpose(-1,-2) + x0[:,:,:,1:-1] @ x0[:,:,:,1:-1].transpose(-1,-2) + xp @ xp.transpose(-1,-2))/3).mean(2)
            cov[:,:,::bsup-binf] = (x0[:,:,:,::bsup-binf] @ x0[:,:,:,::bsup-binf].transpose(-1,-2)).mean(2)

            elt00 = cov + mu @ mu.transpose(-1,-2)
            elt01 = mu
            elt10 = mu.transpose(-1,-2)
            elt11 = torch.ones(batch,len(parts),bsup-binf + 1,1,1)
            output_st.append(torch.cat((torch.cat((elt00,elt01),-1),torch.cat((elt10,elt11),-1)),-2))
        return torch.cat(tuple(output_st),2)


    @staticmethod
    def backward(ctx,grad_output_st):

        input,t0, NP, P = ctx.saved_tensors
        t0, NP , P = int(t0), int(NP), int(P)
        batch,nb_frames,joints,coor,col = input.size()
        grad_input_st = torch.zeros(input.size())
        grad_output_st = grad_output_st.split([len(sequence(nb_frames)[s]) for s in range(6)],2)
        input = input.reshape(batch,nb_frames,joints,coor)
        for s in range(6):
            g = sym(grad_output_st[s]).transpose(1,2)

            binf,bsup = min(sequence(nb_frames)[s]),max(sequence(nb_frames)[s])
            X = input[:,binf:bsup + 1].clone().reshape(batch,bsup-binf + 1,NP, P, coor)
            #outside the edges of frames
            Xs = torch.cat((X[:,1:-1],X[:,:-2],X[:,2:]),-2)
            B = torch.eye(coor + 1,coor).reshape(1,1,1,coor + 1,coor).expand(batch,bsup-binf-1, NP,coor + 1,coor)
            b = torch.cat((torch.zeros(coor),torch.ones(1))).reshape(1,1,1,coor + 1,1).expand(batch,bsup-binf-1,NP,coor + 1,1)
            vect_one = torch.ones(batch,bsup-binf-1, NP, 3 * P,1)
            x = (1/6)*(Xs @ B.transpose(-1,-2) + vect_one @ b.transpose(-1,-2) ) @ g[:,1:-1] @ B
            grad_input_st[:,binf + 1:bsup] += ((x[:,:,:,:P] + x[:,:,:,P:2 * P] + x[:,:,:,2 * P:3 * P])/3).reshape(batch,bsup-binf-1,NP * P,coor).unsqueeze(-1)
            #The edges of frames
            Xs = X[:,::bsup-binf].reshape(batch,2,NP, P,coor)
            B = torch.eye(coor + 1,coor).reshape(1,1,1,coor + 1,coor).expand(batch,2,NP,coor + 1,coor)
            b = torch.cat((torch.zeros(coor),torch.ones(1))).reshape(1,1,1,coor + 1,1).expand(batch,2,NP,coor + 1,1)
            vect_one = torch.ones(batch,2, NP, P,1)
            x = (1/2)*(Xs @ B.transpose(-1,-2) + vect_one @ b.transpose(-1,-2) ) @ g[:,::bsup-binf] @ B
            grad_input_st[:,binf:bsup + 1:bsup-binf] += x.reshape(batch,2, NP * P,coor).unsqueeze(-1)    
        return grad_input_st/3,None, None

class Gauss_agg1_st(nn.Module):
  def __init__(self, parts, t0 = 1):
    super(Gauss_agg1_st,self).__init__()
    self.t0 = t0
    self.parts = parts
  def forward(self,input):
    return Gauss_agg1_st_function.apply(input,self.parts, self.t0)


### ReEig Layer

class ReEig_st_function(Function):
  @staticmethod
  def forward(ctx,input_st,eps):
    eps = torch.tensor(eps)
    #ST
    u,S,v = input_st.svd()
    ctx.save_for_backward(u,S.clone(),eps)
    S[S<eps] = eps
    return u @ S.diag_embed() @ u.transpose(-1,-2),u,S
  
  @staticmethod
  def backward(ctx,grad_output_st,grad_u,grad_S):
    u,S,eps = ctx.saved_tensors
    eps = float(eps)
    #ST
    P = S.unsqueeze(-1).expand(u.size())
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    Q = torch.ones(S.size())
    Q[S<eps] = 0
    Q = Q.diag_embed()
    g = sym(grad_output_st)
    S[S<eps] = eps
    dLdu = 2* g @ u @ S.diag_embed()
    dLdS = Q @ u.transpose(-1,-2) @ g @ u
    idx = torch.arange(0,dLdS.size(3), out = torch.LongTensor())
    k = dLdS[:,:,:,idx,idx].diag_embed()
    grad_input_st = u @ (( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    return grad_input_st,None

class ReEig_st(nn.Module):
  def __init__(self,eps = 10**(-4)):
    super(ReEig_st,self).__init__()
    self.eps = eps

  def forward(self,input_st):
    return ReEig_st_function.apply(input_st,self.eps)


### LogEig Layer

class LogEig_st_function(Function):
  @staticmethod
  def forward(ctx,input_st,u,S):
    #ST
    s = S[:,:,:,:,0].log().diag_embed()
    ctx.save_for_backward(u,S,s)
    return u @ s @ u.transpose(-1,-2)
    
  @staticmethod
  def backward(ctx,grad_output_st):
    u,S,s = ctx.saved_tensors
    g = sym(grad_output_st)
    P = S.clone()
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    dLdu = 2* g @ u @ s
    dLdS = (1/S[:,:,:,:,0]).diag_embed() @ u.transpose(-1,-2) @ g @ u
    idx = torch.arange(0,dLdS.size(3), out = torch.LongTensor())
    k = dLdS[:,:,:,idx,idx].diag_embed()
    grad_input_st = u @(( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    
    return grad_input_st,dLdu,dLdS

class LogEig_st(nn.Module):
  def __init__(self):
    super(LogEig_st,self).__init__()

  def forward(self,input_st,u,S):
    return LogEig_st_function.apply(input_st,u,S.unsqueeze(-1).expand(u.size()))


### VecMat Layer

class VecMat_st_function(Function):

  @staticmethod
  def forward(ctx,input_st):
    ctx.save_for_backward(input_st)
    batch,fingers,nb_frames,row,col = input_st.size()
    input_st.abs_()
    input_st += (sqrt(2)-1)*input_st.triu(1)
    id = torch.LongTensor([[i,j] for i in range(row) for j in range(i,row)]).T
    output_st = input_st[:,:,:,id[0],id[1]].unsqueeze(-1)
    return output_st

  @staticmethod
  def backward(ctx,grad_output_st):
    input_st = ctx.saved_tensors
    input_st = input_st[0]
    batch,fingers,nb_frames,row,col = input_st.size()
    g = torch.zeros(batch,fingers,nb_frames,row,col)
    j = 0
    for i in range(row):
      g[:,:,:,i,i:] = grad_output_st[:,:,:,j:j + row-i,0]
      g[:,:,:,i:,i] = g[:,:,:,i,i:]
      j += row-i
    g += (sqrt(2)-1)*(g.triu(1) + g.tril(-1))
    return g

class VecMat_st(nn.Module):
  def __init__(self):
    super(VecMat_st,self).__init__()

  def forward(self,input_st):
    return VecMat_st_function.apply(input_st)


### Second Gauss aggregation Layer

class Gauss_agg2_st_function(Function):
  @staticmethod
  def forward(ctx,x0,x1,x2,x3,x4,x5, parts):
    ctx.save_for_backward(x0,x1,x2,x3,x4,x5)
    input_st = [x0,x1,x2,x3,x4,x5]
    #ST
    mu = torch.zeros(x0.size(0),6,x0.size(1),x0.size(3),1)
    cov = torch.zeros(x0.size(0),6,x0.size(1),x0.size(3),x0.size(3))
    for s in range(6):
      batch,fingers,nb_frames,row,col = input_st[s].size()
      mu[:,s] = input_st[s].mean(2)
      x = input_st[s]-mu[:,s].unsqueeze(2).expand(batch,fingers,nb_frames,row,col)
      cov[:,s] = (x @ x.transpose(-1,-2)).mean(2)
    elt00 = cov + mu @ mu.transpose(-1,-2)
    elt01 = mu
    elt10 = mu.transpose(-1,-2)
    elt11 = torch.ones(batch,6, len(parts),1,1)
    return torch.cat((torch.cat((elt00,elt01),-1),torch.cat((elt10,elt11),-1)),-2)
  @staticmethod
  def backward(ctx,grad_output_st):
    x0,x1,x2,x3,x4,x5 = ctx.saved_tensors
    input_st = [x0,x1,x2,x3,x4,x5]
    grad_input_st = []
    batch,fingers,nb_frames,row,col = x0.size()
    B = torch.eye(row + 1,row).reshape(1,1,row + 1,row).expand(batch,fingers,row + 1,row)
    b = torch.cat((torch.zeros(row),torch.ones(1))).reshape(1,1,1,row + 1).expand(batch,fingers,1,row + 1)
    g = sym(grad_output_st)
    #ST
    for s in range(6):
      nb_frames = input_st[s].size(2)
      x = input_st[s].squeeze(-1)
      vect_one = torch.ones(batch,fingers,nb_frames,1)
      gr = (2/(nb_frames))* (x @ B.transpose(-1,-2) + vect_one @ b) @ g[:,s] @ B
      grad_input_st.append(gr.unsqueeze(-1))
    return grad_input_st[0],grad_input_st[1],grad_input_st[2],grad_input_st[3],grad_input_st[4],grad_input_st[5], None

class Gauss_agg2_st(nn.Module):
  def __init__(self, parts):
    super(Gauss_agg2_st,self).__init__()
    self.parts = parts
  def forward(self,input_st):
    nb_frames = int(input_st.size(2)/3)
    l_sp = [len(sequence(nb_frames)[s]) for s in range(6)]
    x0,x1,x2,x3,x4,x5 = input_st.split(l_sp,2)
    return Gauss_agg2_st_function.apply(x0,x1,x2,x3,x4,x5, self.parts)


## TS-GA-NET

### First Gauss Aggregation Layer

class Gauss_agg1_ts_function(Function):
  @staticmethod
  def forward(ctx,input_ts, parts, NS):
    NS = torch.tensor(NS)
    NP, P = len(parts), len(parts[0])
    ctx.save_for_backward(input_ts, NS, torch.tensor(NP), torch.tensor(P))
    batch,nb_frames,joints,coordinates,col = input_ts.size()
    #TRY TS
    inputs = input_ts.reshape(batch,nb_frames,NP,P,coordinates,col)
    mu = torch.zeros((batch,6,NP,NS,P,coordinates,1))
    cov = torch.zeros((batch,6,NP,NS, P,coordinates,coordinates))
    for s in range(6):
      binf,bsup = min(sequence(nb_frames)[s]),max(sequence(nb_frames)[s])
      nb_fr = int((bsup-binf + 1)/NS)
      for k in range(NS-1):
        mu[:,s,:,k] = inputs[:,k*nb_fr:(k + 1)*nb_fr].mean(1)
        x = inputs[:,k*nb_fr:(k + 1)*nb_fr]-mu[:,s,:,k].unsqueeze(1).expand(batch,nb_fr,NP, P,coordinates,1)
        cov[:,s,:,k] = (x @ x.transpose(-1,-2)).mean(1)
      k = NS-1
      mu[:,s,:,k] = inputs[:,k*nb_fr:].mean(1)
      x = inputs[:,k*nb_fr:nb_frames]-mu[:,s,:,k].unsqueeze(1).expand(inputs[:,k*nb_fr:nb_frames].size())
      cov[:,s,:,k] = (x @ x.transpose(-1,-2)).mean(1)
    elt00 = cov + mu @ mu.transpose(-1,-2)
    elt01 = mu
    elt10 = mu.transpose(-1,-2)
    elt11 = torch.ones(batch,6,NP,NS, P,1,1)
    return torch.cat((torch.cat((elt00,elt01),-1),torch.cat((elt10,elt11),-1)),-2)

  @staticmethod
  def backward(ctx,grad_output_ts):
    input_ts,NS, NP, P = ctx.saved_tensors
    NS, NP, P = int(NS), int(NP), int(P)
    batch,nb_frames,joints,row,col = input_ts.size()
    grad_input_ts = torch.zeros(input_ts.size())
    inputs = input_ts.transpose(1,2).squeeze().reshape(batch,NP, P,nb_frames,row).type(torch.DoubleTensor)
    #TS
    g = sym(grad_output_ts).type(torch.DoubleTensor)
    B = torch.eye(row + 1,row).reshape(1,1,1,row + 1,row).expand(batch, NP, P,row + 1,row).type(torch.DoubleTensor)
    b = torch.cat((torch.zeros(row),torch.ones(1))).reshape(1,1,1,1,row + 1).expand(batch, NP, P,1,row + 1)
    for s in range(6):
      binf,bsup = min(sequence(nb_frames)[s]),max(sequence(nb_frames)[s])
      nb_fr = int((bsup-binf + 1)/NS)
      vect_one = torch.ones(batch, NP, P,nb_fr,1)
      for k in range(NS-1):
        x = (2/nb_fr)* (inputs[:,:,:,k*nb_fr:(k + 1)*nb_fr] @ B.transpose(-1,-2) + vect_one @ b) @ g[:,s,:,k] @ B
        grad_input_ts[:,k*nb_fr:(k + 1)*nb_fr] += x.reshape(batch, NP * P,nb_fr,row,col).transpose(1,2)
      k = NS-1
      rest_fr = inputs[0,0,0,k*nb_fr:].size(0)
      vect_one = torch.ones(batch, NP, P,rest_fr,1)
      x = (2/nb_fr)* (inputs[:,:,:,k*nb_fr:] @ B.transpose(-1,-2) + vect_one @ b) @ g[:,s,:,k] @ B
      grad_input_ts[:,k*nb_fr:] += x.reshape(batch, NP * P,rest_fr,row,col).transpose(1,2)
    return grad_input_ts/3,None, None

class Gauss_agg1_ts(nn.Module):
  def __init__(self,parts, NS = 15):
    super(Gauss_agg1_ts,self).__init__()
    self.NS = NS
    self.parts = parts
  def forward(self,input):
    return Gauss_agg1_ts_function.apply(input,self.parts, self.NS)


### ReEig Layer

class ReEig_ts_function(Function):
  @staticmethod
  def forward(ctx,input_ts,eps):
    eps = torch.tensor(eps)
    #TS
    u,S,v = input_ts.svd()
    ctx.save_for_backward(u,S.clone(),eps)
    S[S<eps] = eps
    return u @ S.diag_embed() @ u.transpose(-1,-2),u,S 
  
  @staticmethod
  def backward(ctx,grad_output_ts,grad_u,grad_S):
    u,S,eps = ctx.saved_tensors
    #TS
    P = S.unsqueeze(-1).expand(u.size())
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    Q = torch.ones(S.size())
    Q[S<eps] = 0
    Q = Q.diag_embed()
    g = sym(grad_output_ts) 
    S[S<eps] = eps
    dLdu = 2* g @ u @ S.diag_embed()
    dLdS = Q @ u.transpose(-1,-2) @ g @ u
    idx = torch.arange(0,dLdS.size(-1), out = torch.LongTensor())
    k = dLdS[:,:,:,:,:,idx,idx].diag_embed()
    grad_input_ts = u @ (( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    return grad_input_ts,None

class ReEig_ts(nn.Module):
  def __init__(self,eps = 10**(-4)):
    super(ReEig_ts,self).__init__()
    self.eps = eps

  def forward(self,input_ts):
    return ReEig_ts_function.apply(input_ts,self.eps)


### LogEig Layer

class LogEig_ts_function(Function):
  @staticmethod
  def forward(ctx,input_ts,u,S):
    s = S[:,:,:,:,:,:,0].log().diag_embed()
    ctx.save_for_backward(u,S,s)
    return u @ s @ u.transpose(-1,-2) 
    
  @staticmethod
  def backward(ctx,grad_output_ts):
    u,S,s = ctx.saved_tensors
    g = sym(grad_output_ts)
    S[S<0.0001] = 0.0001
    P = S.clone()
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    dLdu = 2* g @ u @ s
    dLdS = (1/S[:,:,:,:,:,:,0]).diag_embed() @ u.transpose(-1,-2) @ g @ u
    idx = torch.arange(0,dLdS.size(-1), out = torch.LongTensor())
    k = dLdS[:,:,:,:,:,idx,idx].diag_embed()
    grad_input_ts = u @(( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    return grad_input_ts,dLdu,dLdS

class LogEig_ts(nn.Module):
  def __init__(self):
    super(LogEig_ts,self).__init__()

  def forward(self,input_ts,u,S):
    return LogEig_ts_function.apply(input_ts,u,S.unsqueeze(-1).expand(u.size()))


### VecMat Layer

class VecMat_ts_function(Function):

  @staticmethod
  def forward(ctx,input_ts):
    ctx.save_for_backward(input_ts)
    #TS
    row = input_ts.size(-1)
    input_ts.abs_()
    input_ts += (sqrt(2)-1)*input_ts.triu(1)
    id = torch.LongTensor([[i,j] for i in range(row) for j in range(i,row)]).T
    output_ts = input_ts[:,:,:,:,:,id[0],id[1]].unsqueeze(-1)
    return output_ts

  @staticmethod
  def backward(ctx,grad_output_ts):
    input_ts = ctx.saved_tensors
    input_ts = input_ts[0]
    #TS
    batch,seq,fingers,NS,joints,row,col = input_ts.size()
    grad_input_ts = torch.zeros(input_ts.size())
    j = 0
    for i in range(row):
      grad_input_ts[:,:,:,:,:,i,i:] = grad_output_ts[:,:,:,:,:,j:j + row-i,0]
      grad_input_ts[:,:,:,:,:,i:,i] = grad_input_ts[:,:,:,:,:,i,i:]
      j += row-i
    grad_input_ts += (sqrt(2)-1)*(grad_input_ts.triu(1) + grad_input_ts.tril(-1))
    return grad_input_ts

class VecMat_ts(nn.Module):
  def __init__(self):
    super(VecMat_ts,self).__init__()

  def forward(self,input_ts):
    return VecMat_ts_function.apply(input_ts)


### Second Gauss aggregation Layer

class Gauss_agg2_ts_function(Function):
  @staticmethod
  def forward(ctx,input_ts):
    ctx.save_for_backward(input_ts)
    #TS
    batch,seq,NP,NS,P,row,col = input_ts.size()
    input_ts = input_ts.reshape(batch,seq,NP, NS * P,row,col)
    mu = input_ts.mean(3)
    x = input_ts-mu.unsqueeze(3).expand(input_ts.size())
    cov = (x @ x.transpose(-1,-2)).mean(3)
    elt00 = cov + mu @ mu.transpose(-1,-2)
    elt01 = mu
    elt10 = mu.transpose(-1,-2)
    elt11 = torch.ones(batch,6, NP,1,1)
    return torch.cat((torch.cat((elt00,elt01),-1),torch.cat((elt10,elt11),-1)),-2)

  @staticmethod
  def backward(ctx,grad_output_ts):
    input_ts = ctx.saved_tensors
    input_ts = input_ts[0]
    #TS
    batch,seq,NP, NS, P,row,col = input_ts.size()
    input_ts = input_ts.reshape(batch,seq, NP, NS * P, row).type(torch.DoubleTensor)
    B = torch.eye(row + 1,row).reshape(1,1,row + 1,row).expand(batch,seq, NP,row + 1,row).type(torch.DoubleTensor)
    b = torch.cat((torch.zeros(row),torch.ones(1))).reshape(1,1,1,row + 1).expand(batch,seq, NP,1,row + 1)
    vect_one = torch.ones(batch,seq, NP,NS* P,1)
    g = sym(grad_output_ts).type(torch.DoubleTensor)
    gr = (2/(NS*4))* (input_ts @ B.transpose(-1,-2) + vect_one @ b) @ g @ B
    grad_input_ts = gr.reshape(batch, seq, NP, NS, P, row, col)    
    return grad_input_ts

class Gauss_agg2_ts(nn.Module):
  def __init__(self):
    super(Gauss_agg2_ts,self).__init__()

  def forward(self,input_ts):
    return Gauss_agg2_ts_function.apply(input_ts)


## SPDC Net

### SPD Aggregation Layer

class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of 
        Stiefel manifold.
    """
    def __new__(cls, data = None, requires_grad = True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad = requires_grad)

    def __repr__(self):
        return self.data.__repr__()


class SPDAgg_function(torch.autograd.Function):
  @staticmethod
  def forward(ctx,input,weights, N):
    ctx.save_for_backward(input,weights, torch.tensor(N))
    output = torch.sum(weights @ input @ (weights.transpose(-1,-2)) ,1 )
    return output

  @staticmethod
  def backward(ctx,grad_output):
    input,weight, N = ctx.saved_tensors
    g = grad_output.unsqueeze(1).expand(input.size(0),int(N),200,200)
    grad_input = weight.transpose(-1,-2) @ g @ weight
    grad_weight = 2* g @ weight @ input
    return grad_input,grad_weight, None

class SPD_Agg(nn.Module):
  def __init__(self, NP, input_size = 56,output_size = 200):
    super(SPD_Agg,self).__init__()
    self.output_size = output_size
    self.input_size = input_size
    self.NP = NP
    self.weight = StiefelParameter(torch.FloatTensor(self.NP,output_size,input_size), requires_grad = True)
    nn.init.orthogonal_(self.weight).requires_grad_()
        
  def forward(self,input):
    weight = self.weight.expand(input.size(0), self.NP,self.output_size,self.input_size)
    return SPDAgg_function.apply(input,weight, self.NP)


### LogEig Layer

class LogEig_spdc_function(torch.autograd.Function):
  @staticmethod
  def forward(ctx,input,vect):
    u,S,v = input.svd()
    ctx.save_for_backward(u,S,torch.tensor(vect))
    output = u @ S.log().diag_embed() @ u.transpose(-1,-2)
    if vect:
      row = output.size(-1)
      output.abs_()
      output += (sqrt(2)-1)*output.triu(1)
      id = torch.LongTensor([[i,j] for i in range(row) for j in range(i,row)]).T
      output = output[:,id[0],id[1]]
    return output

  @staticmethod
  def backward(ctx,grad_output):
    u,S,vect = ctx.saved_tensors
    if vect:
      row = u.size(-2)
      grad_input = torch.zeros(u.size())
      j = 0
      for i in range(row):
        grad_input[:,i,i:] = grad_output[:,j:j + row-i]
        grad_input[:,i:,i] = grad_input[:,i,i:]
        j += row-i
      grad_input += (sqrt(2)-1)*(grad_input.triu(1) + grad_input.tril(-1))
      grad_output = grad_input
    g = sym(grad_output)
    P = S.unsqueeze(-1).expand(u.size())
    P = P - P.transpose(-1,-2)
    mask_zero = torch.abs(P) == 0
    P = 2 / P
    P[mask_zero] = 0
    dLdu = 2* g @ u @ S.log().diag_embed()
    dLdS = (1/S).diag_embed()@ u.transpose(-1,-2) @ g @ u
    idx = torch.arange(0,dLdS.size(-1), out = torch.LongTensor())
    k = dLdS[:,idx,idx].diag_embed()
    grad_input = u @(( P.transpose(-1,-2)*(u.transpose(-1,-2) @ sym(dLdu))) + k) @ u.transpose(-1,-2)
    return grad_input,None

class LogEig_spdc(nn.Module):
  def __init__(self,vect = True):
    super(LogEig_spdc,self).__init__()
    self.vect = vect
  def forward(self,input):
    return LogEig_spdc_function.apply(input,self.vect)
