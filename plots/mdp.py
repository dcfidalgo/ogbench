import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import cmasher as cmr
from tqdm import tqdm
import copy
from functorch import hessian
import os
from einops import einsum, rearrange
import random
import matplotlib as mpl
import torch as th
import itertools
from einops import rearrange

EPSILON = 1e-6

def get_arrow_mask(n=100):
    mask = np.ones((n,n))
    mask = (np.triu(mask, k=n//10)+np.triu(mask, k=2*(n//10)))*(1-np.triu(mask, k=3*(n//10)))
    mask = mask*np.rot90(mask) >= 2
    mask |= np.rot90(mask)
    mask |= np.rot90(mask)
    mask |= np.rot90(mask)
    return mask

def plot_pi_arrows(pi, n=100, s=5, ax=None):
    import itertools
    left, right, up, down = 0, 1, 2, 3

    if ax is None:
        _, ax = plt.subplots(1, 1)

    pi = rearrange(pi, "a (s1 s2) -> s1 s2 a", s1=s)

    pi_repeat = np.zeros((n*s, n*s, 4))

    for (s1, s2) in itertools.product(range(s), range(s)):
        pi_repeat[s1*n:s1*n+n,s2*n:s2*n+n,left] = -np.log(pi[s1, s2, left])
        pi_repeat[s1*n:s1*n+n,s2*n:s2*n+n,right] = -np.log(pi[s1, s2, right])
        pi_repeat[s1*n:s1*n+n,s2*n:s2*n+n,up] = -np.log(pi[s1, s2, up])
        pi_repeat[s1*n:s1*n+n,s2*n:s2*n+n,down] = -np.log(pi[s1, s2, down])

    pi = pi_repeat

    pi_sector = np.zeros_like(pi)

    for (s1,s2) in itertools.product(range(s), range(s)):
        pi_sector[s1*n:s1*n+n,s2*n:s2*n+n,left] = pi[s1*n:s1*n+n,s2*n:s2*n+n,left]*np.rot90(np.triu(np.ones_like(pi[s1*n:s1*n+n,s2*n:s2*n+n,left])))*np.tril(np.ones_like(pi[s1*n:s1*n+n,s2*n:s2*n+n,left]))
        pi_sector[s1*n:s1*n+n,s2*n:s2*n+n,right] = pi[s1*n:s1*n+n,s2*n:s2*n+n,right]*np.triu(np.ones_like(pi[s1*n:s1*n+n,s2*n:s2*n+n,right]))*np.rot90(np.tril(np.ones_like(pi[s1*n:s1*n+n,s2*n:s2*n+n,right])))
        pi_sector[s1*n:s1*n+n,s2*n:s2*n+n,up] = pi[s1*n:s1*n+n,s2*n:s2*n+n,up]*np.triu(np.ones_like(pi[s1*n:s1*n+n,s2*n:s2*n+n,up]))*np.rot90(np.triu(np.ones_like(pi[s1*n:s1*n+n,s2*n:s2*n+n,up])))
        pi_sector[s1*n:s1*n+n,s2*n:s2*n+n,down] = pi[s1*n:s1*n+n,s2*n:s2*n+n,down]*np.rot90(np.tril(np.ones_like(pi[s1*n:s1*n+n,s2*n:s2*n+n,down])))*np.tril(np.ones_like(pi[s1*n:s1*n+n,s2*n:s2*n+n,down]))


    pi_sec_sum = pi_sector.sum(axis=-1)
    mask = np.zeros_like(pi_sec_sum)
    for (s1,s2) in itertools.product(range(5), range(5)):
        mask[s1*n:s1*n+n,s2*n:s2*n+n] = (get_arrow_mask(n) == False)

    masked = np.ma.masked_where(mask, pi_sec_sum)

    ax.imshow(masked, vmin=0, vmax=4, cmap="Grays")

def plot_color_at(cmap: str, pos: tuple, n=100, s=5, ax=None):

    pi_sec_sum = np.ones((n*s, n*s))
    mask = np.zeros_like(pi_sec_sum)
    for (s1,s2) in itertools.product(range(s), range(s)):
        mask[s1*n:s1*n+n,s2*n:s2*n+n] = ((s1,s2) != pos)

    masked = np.ma.masked_where(mask, pi_sec_sum)

    ax.imshow(masked, vmin=0, vmax=4, cmap=cmap)

def negative_conditional_entropy(d):
    """
    Computes the conditional entropy of the policy.
    """
    return th.sum(d * th.log(d/(d.sum(-1) + EPSILON) + EPSILON))

def negative_entropy(d):
    """
    Computes the conditional entropy of the policy.
    """
    return th.sum(d * th.log(d + EPSILON))

def entropy(d):
    """
    Computes the entropy of the policy.
    """
    return -negative_entropy(d)

def jensen_shannon(d_0):
    def jss(d):
        """
        Computes the Jensen-Shannon divergence of the policy w.r.t another policy d_0.
        """
        p = d
        q = d_0
        m = 0.5 * (p + q)
        return 0.5 * (th.sum(p * th.log(p / (m + EPSILON) + EPSILON), dim=(1,2)) + th.sum(q * th.log(q / (m + EPSILON) + EPSILON), dim=(1,2)))
    return jss

def mutual_information(d):
    """
    Computes the Jensen-Shannon divergence between the two policies. 
    This is equivalent to the mutual information I(Z; S, A) when d is the state-action occupancy measure and Z is binary and uniformly distributed.
    """
    p = d[0]
    q = d[1]
    m = 0.5 * (p + q)
    return 0.5 * (th.sum(p * th.log(p / (m + EPSILON) + EPSILON)) + th.sum(q * th.log(q / (m + EPSILON) + EPSILON)))

def get_policy_grid(n_points: int = 100):
    grid = th.stack(th.meshgrid(th.linspace(0., 1., n_points), th.linspace(0., 1., n_points), indexing='xy'))
    grid = grid.reshape(2, -1).T
    return th.stack([grid, 1 - grid], dim=-2)

P_TWO_STATE = th.stack([th.eye(2), 1. - th.eye(2)], dim=-1)

def p_grid_action(n, m):
    """construct a transition matrix for a grid of size n x m"""
    assert n == m  # legacy
    states = n*n
    left = np.diag(np.diag(np.ones((states, states)), -1), -1).astype(int).T
    right = np.diag(np.diag(np.ones((states, states)), 1), 1).astype(int).T
    up = np.diag(np.diag(np.ones((states, states)), -n), -n).astype(int).T
    down = np.diag(np.diag(np.ones((states, states)), n), n).astype(int).T
    for i in range(4):
        left[:, n+n*i] = np.zeros(n*n)
        right[:, (n-1)+n*i] = np.zeros(n*n)
    #up[:, 0:5] = np.zeros(n*n)
    return th.tensor(np.stack([left, right, up, down], axis=-1))

def p_chain(n_states):
    """construct a transition matrix for a chain of n_states"""
    left = np.diag(np.diag(np.ones((n_states, n_states)), -1), -1).astype(int).T
    right = np.diag(np.diag(np.ones((n_states, n_states)), 1), 1).astype(int).T
    return th.tensor(np.stack([left, right], axis=-1))


class FiniteCMDP:
    def __init__(self, 
                 P: th.tensor,
                 r: th.Tensor = None,
                 c: th.Tensor = None,
                 b: th.Tensor = None,
                 mu: th.Tensor = None,
                 gamma: float = 0.9):
        
        self.n_states, self.n_actions = P.size(1), P.size(2)

        self.P = P

        self.r = r if r is not None else th.rand((self.n_states, self.n_actions))
        self.c = c
        self.b = b
        self.mu = mu if mu is not None else th.softmax(th.rand(self.n_states), dim=0)
        self.gamma = gamma

    def P_pi(self, policy):
        return einsum(policy, self.P, "batch aa ss, ss s a -> batch ss aa s a")

    def SR(self, policy):
        P_pi_flat = rearrange(self.P_pi(policy), "batch ss aa s a -> batch (ss aa) (s a)")
        SR_pi_flat = (1-self.gamma)*th.linalg.inv(th.eye(self.n_states*self.n_actions)-self.gamma*P_pi_flat)
        return rearrange(SR_pi_flat, "batch (ss aa) (s a) -> batch ss aa s a", ss=self.n_states, aa=self.n_actions, s=self.n_states, a=self.n_actions)

    def Q_sa(self, policy):
        return einsum(self.SR(policy), self.r, "batch ss aa s a, ss aa -> batch s a")
    
    def QC_sa(self, policy):
        return einsum(self.SR(policy), self.c, "batch ss aa s a, ss aa -> batch s a")

    def V_s(self, policy):
        return einsum(self.Q_sa(policy), policy, "batch s a, batch a s -> batch s")
    
    def VC_s(self, policy):
        return einsum(self.QC_sa(policy), policy, "batch s a, batch a s -> batch s")

    def V(self, policy):
        return einsum(self.V_s(policy), self.mu, "batch s, s -> batch")
    
    def VC(self, policy):
        return einsum(self.VC_s(policy), self.mu, "batch s, s -> batch")
        
    def d_pi(self, policy):
        mu_pi = einsum(self.mu, policy, "s, batch a s -> batch s a")
        return einsum(self.SR(policy), mu_pi, "batch ss aa s a, batch s a -> batch ss aa")

    def rho_pi(self, policy):
        pi_interior = th.clamp(policy, min=EPSILON, max=1-EPSILON)
        return einsum(self.d_pi(policy), th.ones(1, dtype=float), "batch s a, a -> batch s")
    
    def check_shapes(self):
        return {
            "P": self.P.shape,
            "r": self.r.shape,
            "c": self.c.shape,
            "b": self.b.shape,
            "mu": self.mu.shape,
            "gamma": (1,),
        }

policy_grid = get_policy_grid()

class TabularPolicy(th.nn.Module):
    def __init__(self, env: FiniteCMDP, n_policies, init_mode: str = "random", beta=0.1, barrier_mode: str = "log"):
        super().__init__()
        self.env = env
        self.n_states = self.env.n_states
        self.n_actions = self.env.n_actions
        self.n_policies = n_policies
        self.theta = self.init(init_mode)
        self.kappa = th.nn.Parameter(th.ones(1)*0.1)
        self.beta = beta
        self.barrier = lambda x: -th.log(-x) if barrier_mode == "log" else lambda x: 1/(-x)
    
    def init(self, init: str):
        if init == "random":
            self.theta = th.nn.Parameter(th.randn((self.n_policies, self.n_actions, self.n_states)))
        elif init == "uniform":
            theta = th.ones((self.n_policies, self.n_actions, self.n_states))
            self.theta = th.nn.Parameter(theta + th.rand(theta.shape)*0.5)
        elif init == "second":
            theta = th.ones((self.n_policies, self.n_actions, self.n_states))
            theta[:,1,1] += 10
            theta[:,0,0] += 10
            # theta += th.rand(theta.shape)
            self.theta = th.nn.Parameter(th.log(theta))
        elif init == "right":
            theta = th.zeros((self.n_policies, self.n_actions, self.n_states))
            theta[:,1,:] += 1
            self.theta = th.nn.Parameter(th.log(theta))
            
        else:
            raise ValueError("init mode unknown")
        return self.theta
    
    def softmax_policy(self, theta):
        # assuming pi(..., a|s)
        return th.softmax(theta, dim=-2)
    
    def forward(self):
        """
        Forward pass of the policy.
        """
        return self.softmax_policy(self.theta)

    def train(self, potential: callable = None, n_steps=5000, lr=0.1, lagrangian_lr=1.0):

        optimizer = th.optim.SGD([self.theta], lr=lr, momentum=0., dampening=0., weight_decay=0., nesterov=False)
        dual_optimizer = th.optim.SGD([self.kappa], lr=lagrangian_lr, momentum=0., dampening=0., weight_decay=0., nesterov=False)
        policies = []

        for _ in tqdm(range(n_steps)):
            if potential is None:
                if self.env.b is None:  # vanilla policy gradient (no constraints)
                    optimizer.zero_grad()
                    loss = -self.env.V(self.forward())
                    loss.sum().backward()
                    optimizer.step()
                else:  # policy gradient 3 player game (with constraints, see Zahavy et al. 2021)
                    if isinstance(self.env, N_MDP):
                        loss_kappa = -self.kappa*(self.env.g(self.env.d_pi(self.forward())) - self.env.b)
                    else:  # vanilla primal-dual (with constraints)
                        loss_kappa = -self.kappa*(einsum(self.env.c, self.env.d_pi(self.forward()), "b s a, b s a -> b") - self.env.b)
                    dual_optimizer.zero_grad()
                    loss_kappa.sum().backward()
                    dual_optimizer.step()
                    self.kappa.data.clamp_(min=0)
                    # update policy
                    optimizer.zero_grad()
                    if isinstance(self.env, N_MDP):
                        loss = -self.env.V(self.forward()) + self.kappa.clamp(min=0)*(self.env.g(self.env.d_pi(self.forward())) - self.env.b)
                    else:
                        loss = -self.env.V(self.forward()) + self.kappa.clamp(min=0)*(einsum(self.env.c, self.env.d_pi(self.forward()), "b s a, b s a -> b").sum() - self.env.b)
                    loss /= (1 + self.kappa)
                    loss.sum().backward()
                    optimizer.step()
            else: # Hessian policy gradient
                optimizer.zero_grad()
                loss = -self.env.V(self.forward())
                loss.sum().backward()
                J = th.autograd.functional.jacobian(lambda x: self.env.d_pi(self.softmax_policy(x)), self.theta, create_graph=True)
                if self.env.b is None:
                    H = th.autograd.functional.hessian(lambda x: potential(x), self.env.d_pi(self.forward()), create_graph=True)
                else:
                    d = self.env.d_pi(self.forward()).detach()
                    if isinstance(self.env, N_MDP):
                        #c_pi = th.autograd.functional.jacobian(lambda x: self.env.g(x).sum(), d, create_graph=True).detach()
                        H = th.autograd.functional.hessian(lambda x: potential(x) - self.beta*th.log(self.env.g(x).sum()-self.env.b), d, create_graph=True)
                    else:
                        H = th.autograd.functional.hessian(lambda x: potential(x) - self.beta*th.log(self.env.VC(x).sum()-self.env.b), d, create_graph=True)
                J = rearrange(J, "batch ss aa b a s -> (batch ss aa) (b a s)")
                H = rearrange(H, "batch ss aa b s a -> (batch ss aa) (b s a)")
                g = J.transpose(-1, -2)@H@J
                theta_grad = rearrange(self.theta.grad, "batch aa ss -> (batch aa ss)")
                # Solve the linear system H * x = grad
                new_theta_grad = th.linalg.lstsq(g, theta_grad).solution
                self.theta.grad = rearrange(new_theta_grad, "(batch aa ss) -> batch aa ss", batch=self.n_policies, ss=self.n_states, aa=self.n_actions)
                optimizer.step()

            policies.append(self.forward())

        return policies
    
    def belief_state(self, time, noise=0.1):
        """
        Returns the belief state at a given time step.
        """
        P_pi = einsum(self.forward(), self.env.P.float(), "batch a s, ss s a -> batch ss s")
        pt = self.env.mu
        for t in range(time):
            pt = einsum(P_pi, pt, "batch ss s, s -> ss")*(1-noise)
            idx = th.where(pt > 0)[0]
            pt[idx] += noise / 3
            pt[idx+1] += noise / 3
            pt[idx-1] += noise / 3
        return pt

class N_MDP(FiniteCMDP):
    def __init__(self, P, f: callable, g: callable = None, **cmdp_kwargs):
        super().__init__(P=P, **cmdp_kwargs)
        self.f = f
        self.g = g
    
    def Q_sa(self, policy):
        """Compute the expected value of the reward under the policy."""
        r_pi = th.autograd.functional.jacobian(lambda x: self.f(x), self.d_pi(policy), create_graph=True).detach()
        return einsum(self.SR(policy), r_pi, "batch ss aa s a, batch ss aa -> batch s a")
    
    def QC_sa(self, policy):
        """Compute the expected value of the reward under the policy."""
        c_pi = th.autograd.functional.jacobian(lambda x: self.g(x).sum(), self.d_pi(policy), create_graph=True).detach()
        return einsum(self.SR(policy), c_pi, "batch ss aa s a, batch ss aa -> batch s a")
