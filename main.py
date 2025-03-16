# 2017-Proximal Policy Optimization Algorithms （PPO-continue）
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from PPO_continue import PPO_Agent, traj_record
import os, shutil
from datetime import datetime
from Env_Name import Name, BName
import argparse
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def str2bool(V):
    if isinstance(V, bool):
        return V
    elif V.lower in ('yes', 'true', 't', 'y'):
        return True
    elif V.lower in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def eval_func(env_eval, model, eval_seed, e_turns):
    score = 0
    for j in range(e_turns):
        s, _ = env_eval.reset(seed=eval_seed)
        done = False
        while not done:
            a, _, _ = model.action_selection(s, True)
            s_, r, dw, tr, _ = env_eval.step(a)
            done = (dw or tr)
            score += r
            s = s_
    return score / e_turns


parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=4, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
parser.add_argument('--Max_train_steps', type=int, default=int(5e7), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lamda', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
opt = parser.parse_args()
opt.algo = "PPO"
opt.Env_Name = Name[opt.EnvIdex]
opt.BName = BName[opt.EnvIdex]
print(opt)


def main():
    Env_Name = opt.Env_Name
    algo = opt.algo
    bName = opt.BName
    env_train = gym.make(Env_Name, render_mode=None)
    env_eval = gym.make(Env_Name, render_mode="human" if opt.render else None)
    opt.action_dim = env_train.action_space.shape[0]
    opt.a_range = [env_train.action_space.low[0], env_train.action_space.high[0]]
    opt.state_dim = env_train.observation_space.shape[0]
    opt.max_e_steps = env_train._max_episode_steps

    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Algorithm:', algo, '  Env:', bName, '  state_dim:', opt.state_dim, ' control range:', opt.a_range,
          '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(algo, bName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    model = PPO_Agent(**vars(opt))
    if opt.Loadmodel: model.load(BName, opt.ModelIdex)
    traj = traj_record(opt.T_horizon, opt.state_dim, opt.action_dim)

    if opt.render:
        eval_seed = env_seed
        score = eval_func(env_eval, model, eval_seed, 3)
        print(f'Env:{bName}, seed:{eval_seed}, Episode Reward:{score}')
    else:
        total_steps, traj_len = 0, 0
        while total_steps <= opt.Max_train_steps:
            env_seed += 1
            state, _ = env_train.reset(seed=env_seed)
            done = False
            while not done:
                action, prob_a, action_p = model.action_selection(state, False)
                state_next, reward, dw, tr, _ = env_train.step(action)
                done = (dw or tr)

                if opt.EnvIdex == 1 or opt.EnvIdex == 2:
                    if reward <= -100: reward = -1
                elif opt.EnvIdex == 4:
                    reward = (reward + 8) / 8

                traj.add(state, action_p, reward, state_next, done, traj_len, prob_a, dw)
                state = state_next
                traj_len += 1
                total_steps += 1

                if traj_len == opt.T_horizon:
                    model.train(traj)
                    traj_len = 0

                if total_steps % opt.eval_interval == 0 or total_steps == 1:
                    score = eval_func(env_eval, model, opt.seed, 3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:', bName, 'seed:', opt.seed,
                          'steps: {}k'.format(int(total_steps / 1000)), 'score:', score)

                if total_steps % opt.save_interval == 0:
                    model.save(bName, total_steps)

    env_show = gym.make(Env_Name, render_mode="human")
    score = eval_func(env_show, model, env_seed, 10)
    print(f'Env:{bName}, seed:{env_seed}, Episode Reward:{score}')
    env_show.close()
    env_train.close()
    env_eval.close()


if __name__ == '__main__':
    main()
