import torch
from torch.optim.optimizer import Optimizer


class IFSGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        """

        自定义SGDM优化器
        :param params: 待优化的参数
        :param lr: 学习率
        :param momentum: 动量参数
        :param weight_decay: 权重衰减（L2正则化）
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
        super(IFSGD, self).__init__(params, defaults)

        self._int_epoch = 1

    @torch.no_grad()
    def step(self, closure=None, epoch=None):
        """
        执行优化的一步
        :param closure: 闭包函数，用于重新评估模型
        """

        if epoch is None:
            raise ValueError('epoch不能为None！')
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None:
                    continue

                d_p = param.grad.data

                # 应用权重衰减
                if weight_decay != 0:
                    d_p.add_(param.data, alpha=weight_decay)

                # 如果动量缓冲区不存在，则初始化
                if 'momentum_buffer' not in self.state[param]:
                    self.state[param]['momentum_buffer'] = torch.clone(d_p).detach()
                    self.state[param]['integral_moment'] = torch.clone(d_p).detach()
                else:
                    self.state[param]['momentum_buffer'].mul_(momentum).add_(d_p)
                    if self._int_epoch > epoch:
                        self.state[param]['integral_moment'].mul_(0.1).add_(self.state[param]['momentum_buffer'])

                if self._int_epoch > epoch:
                    param.data.add_(-lr ** 2, self.state[param]['integral_moment'])
                else:
                    param.data.add_(-lr, self.state[param]['momentum_buffer'])

        return loss
