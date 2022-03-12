import torch
from torch import nn

def retraction(A, ref):
    data = A + ref
    Q, R = data.qr()
    # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
    sign = (R.triu().tril().sign() + 0.5).sign().triu().tril()
    out = Q @ sign
    return out

def orthogonal_projection(A, B):
    out = A - A @ B.transpose(-1,-2) @ B
    return out

# Optimizer
class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of 
        Stiefel manifold.
    """
    def __new__(cls, data = None, requires_grad = True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad = requires_grad)

    def __repr__(self):
        return self.data.__repr__()

class StiefelMetaOptimizer(object):
    """This is a meta optimizer which uses other optimizers for updating parameters
        and remap all StiefelParameter parameters to Stiefel space after they have been updated.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    print("none")
                    continue
                if isinstance(p, StiefelParameter):
                    if id(p) not in self.state:
                        self.state[id(p)] = p.data.clone()
                    else:
                        self.state[id(p)].fill_(0).add_(p.data)
                    #p.data.fill_(0)
                    trans = orthogonal_projection(p.grad.data, p.data)
                    p.grad.data.fill_(0).add_(trans)
                    
        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    print("none")
                    continue
                if isinstance(p, StiefelParameter):
                    trans = retraction(-0.1* p.grad.data, self.state[id(p)])
                    p.data.fill_(0).add_(trans)
        return loss
