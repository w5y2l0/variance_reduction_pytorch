import math
import torch
import itertools as it
from torch.optim import Optimizer
from collections import defaultdict

class Lookahead(Optimizer):
    '''
    PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    '''
    def __init__(self, optimizer,alpha=0.5, k=6,pullback_momentum="none"):
        '''
        :param optimizer:inner optimizer
        :param k (int): number of lookahead steps
        :param alpha(float): linear interpolation factor. 1.0 recovers the inner optimizer.
        :param pullback_momentum (str): change to inner optimizer momentum on interpolation update
        '''
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'k':self.k,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(1.0 - self.alpha, param_state['cached_params'])  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(
                            1.0 - self.alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss
class STORM(Optimizer):
    r"""Implements STORM algorithm.

    It has been proposed in `Momentum-Based Variance Reduction in Non-Convex SGD`_.
    ... Momentum-Based Variance Reduction in Non-Convex SGD:
        https://arxiv.org/abs/1905.10018

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        k (float, optional): hyperparameter as described in paper
        w (float, optional): hyperparameter as described in paper
        c (float, optional): hyperparameter to be swept over logarithmically spaced grid as per paper
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, k=0.1, w=0.1, c=1, weight_decay=0):
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(k=k, w=w, c=c, weight_decay=weight_decay)
        super(STORM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(STORM, self).__setstate__(state)

    # Performs a single optimization step 
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('STORM does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # No. of steps
                    state['step'] = 0
                    # Learning rate (given as 'η(t)' in paper for step t)
                    state['lr'] = group['k']/group['w']**(1/3)
                    # Square of gradients (given as 'G(t)^2' in paper for step t)
                    state['G^2'] = 0
                    # Momentum (given as 'd(t)' in paper for step t)
                    state['d'] = 0
                    # Gradients before optimization step (given as G(t-1) in paper for step t)
                    state['prev_grad'] = 0
                    # Given as 'c.η(t)^2' in paper for step t
                    state['a'] = 0
                
                # Retrieving variables
                grad_sqr_sum, d, learning_rate,prev_grad,a = state['G^2'], state['d'], state['lr'], state['prev_grad'],state['a']
                k, w, c = group['k'], group['w'], group['c']
                state['step'] += 1

                # weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Change in state for this optimization step
                grad_sqr_sum += (torch.norm(grad).item())**2
                learning_rate = k/(w + grad_sqr_sum)**(1/3)
                d = grad + (1-a)*(d-prev_grad)

                # Data update step
                p.data = p.data - learning_rate*d
                
                # Change in state for next optimization step
                a = c*(learning_rate**2)
                prev_grad = grad

        return loss
