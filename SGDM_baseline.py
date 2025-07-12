import torch
from torch.optim import Optimizer

class SGDM_baseline(Optimizer):
    def __init__(self, params, lr=1e-1, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SGDM_baseline, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get the gradient and parameter data
                grad = p.grad.data

                # Weight decay (L2 regularization)
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Initialize momentum buffer
                if 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    self.state[p]['momentum_buffer'].mul_(group["momentum"]).add_(grad)

                # Parameter update
                p.add_(-group['lr'], self.state[p]['momentum_buffer'])

        return loss
