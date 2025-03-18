import torch
import torch.nn.functional as F
import numpy as np
import copy
import math
from Net_con import AC_net_beta

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.ACNet = AC_net_beta(self.action_dim, self.state_dim, self.net_width).to(device)
        self.optimizer = torch.optim.Adam(self.ACNet.parameters(), lr=self.lr)

    def action_selection(self, s, iseval):
        s = torch.FloatTensor(s).to(device)
        with torch.no_grad():
            if iseval:
                a_p = self.ACNet.get_action(s)
            else:
                Distri = self.ACNet.get_distri(s)
                a_p = Distri.sample()
            prob_a = None if iseval else Distri.log_prob(a_p).cpu().numpy()
            Vs = None if iseval else self.ACNet.get_critic(s).cpu().numpy()
            return a_p.cpu().numpy(), prob_a, Vs

    def train(self, traj):
        self.entropy_coef *= self.entropy_coef_decay
        s, a, r, s_, done, prob_old, dw, vs = traj.read()
        traj_len = len(r)
        t_turns = math.ceil(traj_len / self.batch_size)
        with torch.no_grad():
            V = self.ACNet.get_critic(s)
            V_ = self.ACNet.get_critic(s_)
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
            Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-5)

        for t in range(self.K_epochs):
            Ind = np.arange(traj_len)
            np.random.shuffle(Ind)
            s, a, V_target, prob_old, Adv, vs = s[Ind].clone(), a[Ind].clone(), V_target[Ind].clone(), \
                prob_old[Ind].clone(), Adv[Ind].clone(), vs[Ind].clone()
            for j in range(t_turns):
                Ind_batch = slice(j * self.batch_size, min(traj_len, (j + 1) * self.batch_size))
                Distri = self.ACNet.get_distri(s[Ind_batch])
                prob_a = Distri.log_prob(a[Ind_batch])

                r_t = torch.exp(prob_a.sum(-1, keepdim=True) - prob_old[Ind_batch].sum(-1, keepdim=True))
                L1 = r_t * Adv[Ind_batch]
                L2 = torch.clip(r_t, 1 - self.clip_rate, 1 + self.clip_rate) * Adv[Ind_batch]
                E_Distri = Distri.entropy().sum(-1, keepdim=True)
                # E_Distri = (torch.exp(prob_old[Ind_batch])*(prob_old[Ind_batch]-prob_a)).sum(-1, keepdim=True)
                E_loss = E_Distri.mean()
                A_loss = -torch.mean(torch.min(L1, L2))

                if self.value_clip:
                    vs_now = self.ACNet.get_critic(s[Ind_batch])
                    C_L1 = (vs_now - V_target[Ind_batch])**2
                    C_clip = torch.clip(vs_now, vs[Ind_batch] - self.clip_rate, vs[Ind_batch] + self.clip_rate)
                    C_L2 = (C_clip - V_target[Ind_batch])**2
                    C_loss = 0.5 * torch.max(C_L1, C_L2).mean()
                else:
                    C_loss = 0.5 * F.mse_loss(self.ACNet.get_critic(s[Ind_batch]), V_target[Ind_batch])
                loss = A_loss - self.entropy_coef * E_loss + C_loss * self.critic_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ACNet.parameters(), 40)
                self.optimizer.step()

    def save(self, EnvName, timestep):
        torch.save(self.ACNet.state_dict(), "./model/{}_AC{}.pth".format(EnvName, timestep))

    def load(self, EnvName, timestep):
        self.ACNet.load_state_dict(
            torch.load("./model/{}_AC{}.pth".format(EnvName, timestep), map_location=device))


class traj_record(object):
    def __init__(self, T_horizon, state_n, action_n):
        self.s = np.zeros((T_horizon, state_n), dtype=np.float32)
        self.s_ = np.zeros((T_horizon, state_n), dtype=np.float32)
        self.a = np.zeros((T_horizon, action_n), dtype=np.float32)
        self.r = np.zeros((T_horizon, 1), dtype=np.float32)
        self.done = np.zeros((T_horizon, 1), dtype=np.bool_)
        self.dw = np.zeros((T_horizon, 1), dtype=np.bool_)
        self.prob = np.zeros((T_horizon, action_n), dtype=np.float32)
        self.Vs = np.zeros((T_horizon, 1), dtype=np.float32)

    def add(self, s, a, r, s_, done, Ind, prob_a, dw, Vs):
        self.s[Ind] = s
        self.a[Ind] = a
        self.r[Ind] = r
        self.s_[Ind] = s_
        self.done[Ind] = done
        self.prob[Ind] = prob_a
        self.dw[Ind] = dw
        self.Vs[Ind] = Vs

    def read(self):
        return (
            torch.FloatTensor(self.s).to(device),
            torch.FloatTensor(self.a).to(device),
            torch.FloatTensor(self.r).to(device),
            torch.FloatTensor(self.s_).to(device),
            torch.BoolTensor(self.done).to(device),
            torch.FloatTensor(self.prob).to(device),
            torch.BoolTensor(self.dw).to(device),
            torch.FloatTensor(self.Vs).to(device)
        )
