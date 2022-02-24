import random
import numpy as np
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from agent import Agent
from environment import Environment


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dqn(env, agent, n_episodes=2000, max_t=1000, eps=1.0, eps_end=0.01, eps_decay=0.995, print_freq=100):
    """
    Deep Q-Learning
    :param agent: agent.Agent class instance
    :param env: environment class instance compatible with OpenAI gym
    :param n_episodes: (int) maximum number of training episodes
    :param max_t: (int) maximum number of timesteps per episode
    :param eps: (float) starting value of epsilon, for epsilon-greedy action selection
    :param eps_end: (float) minimum value of epsilon
    :param eps_decay: (float) multiplicative factor (per episode) for decreasing epsilon
    :param print_freq: (int) print frequency of episodic score
    :return: scores: (list[float]) scores of last 100 episodes
    """
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset(options={'train_mode': True})
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            experience_dict = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
            agent.step(experience_dict)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-print_freq:]):.2f}\tMax Score: {np.max(scores[-print_freq:])}\tMin Score: {np.min(scores[-print_freq:])}\tEps: {eps:.3f}\tTstep: {agent.t_step}',
              end="\n" if i_episode % print_freq == 0 else "")
        eps = max(eps_end, eps_decay * eps)
    env.reset(options={'train_mode': True})
    return scores


if __name__ == '__main__':
    """
    * lr: 5e-4 (learning rate for updating local q-network)
* gamma: 0.9 (discount factor when calculating return)
* tau: 1e-3 (learning rate for updating target q-network)
* update_freq: 4 (update local/target q-network every ~ steps)
* buffer_size: 1e5 (maximum number of experiences to save in replay buffer)
* batch_size: 64 (number of experiences to do one step of update)
* n_episodes: 1000 (total number of episodes to play)
* max_t: 1000 (maximum steps to take in single episode)
* eps: 1 (starting epsilon for epsilon-greedy)
* eps_end: 0.01 (minimum value for epsilon)
* eps_decay: 0.995 (decay rate per step for epsilon)
    """
    parser = ArgumentParser()
    parser.add_argument('--env_path', type=str, default='Banana_Linux/Banana.x86_64', help='path for unity environment')
    parser.add_argument('--save_path', type=str, default='qnetwork.ckpt', help='save path for trained q-network weights')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for updating local q-network')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor when calculating return')
    parser.add_argument('--tau', type=float, default=1e-3, help='learning rate for updating target q-network')
    parser.add_argument('--update_freq', type=int, default=4, help='update local/target q-network every ~ steps')
    parser.add_argument('--buffer_size', type=int, default=int(1e5), help='maximum number of experiences to save in replay buffer')
    parser.add_argument('--batch_size', type=int, default=64, help='number of experiences to do one step of update')
    parser.add_argument('--n_episodes', type=int, default=1000, help='total number of episodes to play')
    parser.add_argument('--max_t', type=int, default=1000, help='maximum steps to take in single episode')
    parser.add_argument('--eps', type=float, default=1., help='starting epsilon for epsilon-greedy')
    parser.add_argument('--eps_end', type=float, default=0.01, help='minimum value for epsilon')
    parser.add_argument('--eps_decay', type=float, default=0.995, help='decay rate per step for epsilon')
    parser.add_argument('--print_freq', type=int, default=100, help='print training status every ~ steps')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    seed_all(args.seed)

    env = Environment(args.env_path)

    agent = Agent(
        state_size=env.observation_size[0],
        action_size=env.action_size,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        update_freq=args.update_freq,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size
    )

    scores = dqn(
        env=env,
        agent=agent,
        n_episodes=args.n_episodes,
        max_t=args.max_t,
        eps=args.eps,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        print_freq=args.print_freq
    )

    torch.save(agent.qnetwork_local.state_dict(), args.save_path)

    plt.figure(figsize=(20, 20))
    plt.plot(scores)
    plt.title(f'MAX{np.max(scores)} / LAST{scores[-1]}')
    plt.show()
