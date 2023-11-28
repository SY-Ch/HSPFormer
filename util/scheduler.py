
import os
import math
import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, warmup_iters, epoch_iters , warmup_type='linear', warmup_ratio=1e-6, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.warmup_iters = warmup_iters * epoch_iters
        self.warmup_type = warmup_type
        self.eta_min = eta_min
        self.warmup_ratio = warmup_ratio
        self.epoch = (T_max - warmup_iters ) * epoch_iters
        self.optimizer = optimizer
        self.regular_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        super(WarmupCosineAnnealingLR, self).__init__(optimizer,T_max=self.epoch,eta_min=eta_min,last_epoch= last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            if self.warmup_type == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
            elif self.warmup_type == 'linear':
                k = (1 - self.last_epoch / self.warmup_iters) * (1 -self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
            elif self.warmup_type == 'exp':
                k = self.warmup_ratio**(1 - self.last_epoch / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in self.regular_lr]
            else:
                raise ValueError(f"Invalid warmup type: {self.warmup_type}")
            return warmup_lr
        else:
            lr = super(WarmupCosineAnnealingLR, self).get_lr()
        return lr


class WarmupLambdaLR(LambdaLR):
    def __init__(self, optimizer, warmup_iters, warmup_type, lambda_func, warmup_ratio=1e-6, last_epoch=-1):
        self.warmup_type = warmup_type
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.regular_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        super(WarmupLambdaLR, self).__init__(optimizer, lambda_func, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            if self.warmup_type == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
            elif self.warmup_type == 'linear':
                k = (1 - self.last_epoch / self.warmup_iters) * (1 -self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
            elif self.warmup_type == 'exp':
                k = self.warmup_ratio**(1 - self.last_epoch / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in self.regular_lr]
            else:
                raise ValueError(f"Invalid warmup type: {self.warmup_type}")
            return warmup_lr
        else:
            return super(WarmupLambdaLR, self).get_lr()



class WarmupPolyLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs, warmup_type='linear', warmup_iters=5, warmup_ratio=1e-6, power=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        self.warmup_iters = warmup_iters
        self.warmup_type = warmup_type
        self.warmup_ratio = warmup_ratio
        self.regular_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            if self.warmup_type == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
            elif self.warmup_type == 'linear':
                k = (1 - self.last_epoch / self.warmup_iters) * (1 -self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
            elif self.warmup_type == 'exp':
                k = self.warmup_ratio**(1 - self.last_epoch / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in self.regular_lr]
            else:
                raise ValueError(f"Invalid warmup type: {self.warmup_type}")
            return warmup_lr
        else:
            decay_factor = (1 - self.last_epoch / self.max_epochs) ** self.power
            return [base_lr * decay_factor for base_lr in self.base_lrs]

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, power=0.9, last_epoch=-1) -> None:
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.power
        return [factor*lr for lr in self.base_lrs]
    

def get_scheduler(optimizer, opt):
    if opt.lr_warmup == '':
        if opt.lr_policy == 'lambda':
            lambda_rule = lambda epoch: opt.lr_gamma**(
                (epoch + 1) // opt.lr_decay_epochs)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                            step_size=opt.lr_decay_iters,
                                            gamma=0.1)
        elif opt.lr_policy == 'poly':
            scheduler = PolyLR(optimizer, opt.nepoch, opt.lr_power)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=opt.nepoch * math.floor(opt.total_samples / int(os.environ['GPUS_PER_NODE'])))
        elif opt.lr_policy == 'OneCycle':
             scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=2e-4,
                                                 epochs=opt.nepoch,steps_per_epoch=math.floor(opt.total_samples / int(os.environ['GPUS_PER_NODE'])),
                                                 pct_start=0.009375,anneal_strategy='cos')
        print("Current learning rate scheduler type: {}".format(opt.lr_policy))
    elif opt.lr_warmup != '':
        if opt.lr_policy == 'poly':
            scheduler = WarmupPolyLR(optimizer, opt.nepoch* math.floor(opt.total_samples), opt.lr_warmup,
                                     warmup_iters = opt.warmup_epoch * math.floor(opt.total_samples), power = opt.lr_power,warmup_ratio = opt.warmup_ratio)
        elif opt.lr_policy == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, T_max=opt.nepoch * math.floor(opt.total_samples / int(os.environ['GPUS_PER_NODE'])),
                                                warmup_iters = opt.warmup_epoch * math.floor(opt.total_samples / int(os.environ['GPUS_PER_NODE'])),
                                                warmup_type = opt.lr_warmup,
                                                warmup_ratio = opt.warmup_ratio)
        elif opt.lr_policy == 'lambda':
            lambda_rule = lambda epoch: opt.lr_gamma**(
                (epoch + 1) // opt.lr_decay_epochs)
            scheduler = WarmupLambdaLR(optimizer, opt.warmup_epoch,
                                       opt.lr_warmup, lambda_rule)

        print("Current learning rate scheduler type: {} with warm-up ({})".
              format(opt.lr_policy, opt.lr_warmup))
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler