
import torch


class SimpleSGD(torch.optim.Optimizer):

    def __init__(self, parameters, lr=1):
        self.lr = lr
        defaults = dict(lr=lr)
        super(SimpleSGD, self).__init__(parameters, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(-self.lr, p.grad.data)

    def update(self, factor=2):
        self.lr /= factor


