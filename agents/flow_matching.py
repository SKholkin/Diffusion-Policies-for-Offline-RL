
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import (extract,
                            Losses)


class FlowMatching(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True):
        super(FlowMatching, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Don't forget clamping the by max action bounds!!
        self.max_action = max_action

        # vector net
        self.model = model

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.loss_fn = Losses[loss_type]()
    
        self.invariant_dist = torch.distributions.normal.Normal(0, 1)

    # ------------------------------------------ sampling ------------------------------------------#
    
    @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        # Solving ODE by Euler steps
        batch_size = state.shape[0]

        x_0 = self.invariant_dist.sample([batch_size, self.action_dim]).to(state.device)
        x_t = x_0

        euler_dt = 1 / self.n_timesteps
        t_range = torch.arange(0, 1, step=euler_dt, device=x_t.device)

        for t in t_range:
            t = t.reshape([1]).repeat([batch_size])
            # must be N X D_action, N, N X D_state

            x_t = x_t + self.model(x_t, t, state) * euler_dt

        return x_t.clamp_(-self.max_action, self.max_action)  

    # ------------------------------------------ training ------------------------------------------#

    def loss(self, x_1, state, weights=1.0):
        # x_1: action
        batch_size = len(x_1)
        t = torch.rand((batch_size, ), device=x_1.device)

        # x_1: Data
        # x_0: Gaussian
        x_0 = self.invariant_dist.sample(x_1.shape).to(x_1.device)

        t = t.reshape([-1, 1])
        x_t = t * x_1 + (1 - t) * x_0

        t = t.squeeze()
        x_t_hat = self.model(x_t, t, state)

        loss = torch.norm((x_t_hat - (x_1 - x_0)).reshape([batch_size, -1]), dim=-1)
        return loss.mean()

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)
