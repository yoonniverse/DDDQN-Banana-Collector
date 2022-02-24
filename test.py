from argparse import ArgumentParser
import time
import torch
from environment import Environment
from agent import Agent
from train import seed_all


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env_path', type=str, default='Banana_Linux/Banana.x86_64')
    parser.add_argument('--model_path', type=str, default='qnetwork.ckpt')
    parser.add_argument('--n_episodes', type=int, default=3)
    args = parser.parse_args()

    seed_all(0)

    env = Environment(args.env_path)

    agent = Agent(state_size=env.observation_size[0], action_size=env.action_size)
    agent.qnetwork_local.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    mean_score = 0
    for i in range(args.n_episodes):
        seed_all(i)
        state = env.reset(options={'train_mode': False})
        score = 0
        while True:
            action = agent.act(state, eps=0)
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        mean_score += score / args.n_episodes
        print(f'{i}th test score: {score}')
    print(f'MEAN TEST SCORE: {mean_score}')
