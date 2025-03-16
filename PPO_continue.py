import torch
import torch.nn.functional as F
import numpy as np
import copy
import math
from Net_con import A_net_beta, C_net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# line
def mapping(a, a_range):
    a = (a_range[1] - a_range[0]) * a + a_range[0]
    return a


class PPO_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.Actor = A_net_beta(self.action_dim, self.state_dim, self.net_width).to(device)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.a_lr)
        self.Critic = C_net(self.state_dim, self.net_width).to(device)
        self.C_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.c_lr)

    def action_selection(self, s, iseval):
        s = torch.FloatTensor(s).to(device)
        with torch.no_grad():
            if iseval:
                a_p = self.Actor.get_action(s)
            else:
                Distri = self.Actor.get_distri(s)
                a_p = Distri.sample()
            prob_a = None if iseval else Distri.log_prob(a_p).cpu().numpy()
            a = torch.clip(a_p, 0, 1)
            a = mapping(a, self.a_range)
            a_p = a_p.cpu().numpy()
            return a.cpu().numpy(), prob_a, a_p

    def train(self, traj):
        self.entropy_coef *= self.entropy_coef_decay
        s, a, r, s_, done, prob_old, dw = traj.read()
        traj_len = len(r)
        t_turns = math.ceil(traj_len / self.batch_size)
        with torch.no_grad():
            V = self.Critic(s)
            V_ = self.Critic(s_)
            delta = r + ~dw * self.gamma * V_ - V
            delta = delta.cpu().view(1, -1).squeeze(0).numpy()
            Adv = [0]

            done = done.cpu().view(1, -1).squeeze(0).numpy()
            for j in range(traj_len - 1, -1, -1):
                A = delta[j] + self.gamma * self.lamda * Adv[-1] * (~done[j])
                Adv.append(A)
            Adv = Adv[::-1]
            Adv = copy.deepcopy(Adv[0:-1])
            Adv = torch.FloatTensor(Adv).unsqueeze(-1).to(device)
            V_target = Adv + V

        for t in range(self.K_epochs):
            Ind = np.arange(traj_len)
            np.random.shuffle(Ind)
            s, a, r, V_target, prob_old, Adv = s[Ind].clone(), a[Ind].clone(), r[Ind].clone(), V_target[Ind].clone(), prob_old[
                Ind].clone(), Adv[Ind].clone()
            for j in range(t_turns):
                Ind_batch = slice(j * self.batch_size, min(traj_len, (j + 1) * self.batch_size))
                Distri = self.Actor.get_distri(s[Ind_batch])
                prob_a = Distri.log_prob(a[Ind_batch])

                r_t = torch.exp(prob_a.sum(-1, keepdim=True) - prob_old[Ind_batch].sum(-1, keepdim=True))
                L1 = r_t * Adv[Ind_batch]
                L2 = torch.clip(r_t, 1 - self.clip_rate, 1 + self.clip_rate) * Adv[Ind_batch]
                E_Distri = Distri.entropy().sum(-1, keepdim=True)
                A_loss = -torch.mean(torch.min(L1, L2) + E_Distri*self.entropy_coef)

                self.A_optimizer.zero_grad()
                A_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), 40)
                self.A_optimizer.step()

                C_loss = F.mse_loss(self.Critic(s[Ind_batch]), V_target[Ind_batch])
                for name, param in self.Critic.named_parameters():
                    if 'weight' in name:
                        C_loss += param.pow(2).sum() * self.l2_reg
                self.C_optimizer.zero_grad()
                C_loss.backward()
                self.C_optimizer.step()

    def save(self, EnvName, timestep):
        torch.save(self.Actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName, timestep))
        torch.save(self.Critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName, timestep))

    def load(self, EnvName, timestep):
        self.Actor.load_state_dict(
            torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=device))
        self.Critic.load_state_dict(
            torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=device))


class traj_record(object):
    def __init__(self, T_horizon, state_n, action_n):
        self.s = np.zeros((T_horizon, state_n), dtype=np.float32)
        self.s_ = np.zeros((T_horizon, state_n), dtype=np.float32)
        self.a = np.zeros((T_horizon, action_n), dtype=np.float32)
        self.r = np.zeros((T_horizon, 1), dtype=np.float32)
        self.done = np.zeros((T_horizon, 1), dtype=np.bool_)
        self.dw = np.zeros((T_horizon, 1), dtype=np.bool_)
        self.prob = np.zeros((T_horizon, action_n), dtype=np.float32)

    def add(self, s, a, r, s_, done, Ind, prob_a, dw):
        self.s[Ind] = s
        self.a[Ind] = a
        self.r[Ind] = r
        self.s_[Ind] = s_
        self.done[Ind] = done
        self.prob[Ind] = prob_a
        self.dw[Ind] = dw

    def read(self):
        return (
            torch.FloatTensor(self.s).to(device),
            torch.FloatTensor(self.a).to(device),
            torch.FloatTensor(self.r).to(device),
            torch.FloatTensor(self.s_).to(device),
            torch.BoolTensor(self.done).to(device),
            torch.FloatTensor(self.prob).to(device),
            torch.BoolTensor(self.dw).to(device)
        )
