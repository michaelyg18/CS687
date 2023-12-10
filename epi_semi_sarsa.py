import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
import torch
import torch.nn as nn


# neural network to approximate action-value
class ActionValueApproximator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, (input_dim + 1) * 3),
            nn.ReLU(),
            nn.Linear((input_dim + 1) * 3, (input_dim + 1) * 2),
            nn.ReLU(),
            nn.Linear((input_dim + 1) * 2, (input_dim + 1)),
            nn.ReLU(),
            nn.Linear((input_dim + 1), 1)
        )

    def forward(self, x):
        return self.mlp(x)

# q_func is a neural network that returns a prediction
def episodic_semigradient_nstep_sarsa(env, q_func, alpha, gamma, epsilon, n, num_episodes):
    # initialize value-function weights arbitrarily
    loss_func = nn.MSELoss()  # does reduction matter?
    optimizer = torch.optim.Adam(q_func.parameters(), lr=alpha)

    # all store and access operations (S, A, R) can take index mod n+1?

    # loop for each episode
    for episode in range(1, num_episodes + 1):
        print("Episode ", episode)
        # initialize and store non-terminal S0
        state, _ = env.reset()
        states = [state]
        # states = [0] * (n + 1)

        # select and store action A0 ~ pi(·|S0), or epsilon greedy wrt q(S0, ·, w)
        action = epsilon_greedy(env, state, q_func, epsilon)
        actions = [action]
        rewards = [0]

        # loop for t = 0, 1, 2...
        t = 0
        T_cap = np.inf
        while True:
            if t < T_cap:
                # take action At
                # observe and store next reward as Rt+1, and the next state as St+1
                next_state, reward, terminated, truncated, info = env.step(action)
                states.append(next_state)
                rewards.append(reward)

                # if St+1 is terminal, then:
                if terminated or truncated:
                    T_cap = t + 1
                else:
                    # select and store At+1 ~ pi(·|St+1) or epsilon greedy wrt q(St+1, ·, w)
                    next_action = epsilon_greedy(env, next_state, q_func, epsilon/(t+1))
                    actions.append(next_action)

            # tau is time whose estimate is being updated
            tau = t - n + 1
            if tau >= 0:
                i = tau + 1
                sum_lim = min(tau + n, T_cap)
                # summation formula: take slice of reward list, multiply in powers of gamma
                G = np.sum(np.multiply(rewards[i:sum_lim + 1], np.power(gamma, np.arange(sum_lim - i + 1))))
                G = torch.tensor([G])

                if tau + n < T_cap:
                    # here, pass state and action at time tau+n to q_func, get a prediction back (forward pass)
                    state_action_tau_n = torch.tensor(np.append(states[tau+n], actions[tau+n]))
                    q_pred = q_func(state_action_tau_n)
                    # G += q_func(state_action_tau_n)
                    G += q_pred

                # weight update
                state_action_tau = torch.tensor(np.append(states[tau], actions[tau]))
                q_pred = q_func(state_action_tau)
                # MSE of St and Ut
                # print(G)
                loss = loss_func(q_pred, G)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if tau == T_cap - 1:
                break

            # next timestep
            t += 1

    return policy


def epsilon_greedy(env, state, q_func, epsilon):
    # pick greedy with 1-epsilon chance
    if random.random() > epsilon:
        # decide action with softmax loss?
        # for every possible action in current state, get q value and take argmax
        q_vals = []
        for action in range(env.action_space.n):
            q_pred = q_func(torch.tensor(np.append(state, action))).detach().numpy()
            q_vals.append(q_pred)
        return np.argmax(q_vals)
    # explore with epsilon chance
    else:
        # random action in action space (n choices in discrete case)
        return env.action_space.sample()


if __name__ == '__main__':

    # mountain car MDP
    # action space: Discrete(3) (left, no accel, right)
    # state space: Box([-1.2 -0.07], [0.6 0.07], (2,), float32)
    env = gym.make('MountainCar-v0', render_mode="human")
    # print(env._max_episode_steps)  200 by default
    env._max_episode_steps = 1000
    # env = gym.make('LunarLander-v2', render_mode="human")
    observation, info = env.reset(seed=42)  # set RNG generation seed for future episodes

    # initialization
    input_dim = observation.shape[0] + 1  # input dimension to approximator network is state space + the action taken
    q_func = ActionValueApproximator(input_dim)
    q_func.double()
    # q_func.int()
    alpha = 0.3
    gamma = .99
    epsilon = 1
    n = 4
    num_episodes = 50

    episodic_semigradient_nstep_sarsa(env, q_func, alpha, gamma, epsilon, n, num_episodes)
    # for _ in range(10):
    #     act = env.action_space.sample()  # agent policy that uses the observation and info
    #     observation, reward, terminated, truncated, info = env.step(act)
    #     # print(observation, act)
    #     # print(torch.tensor(np.append(observation, act)))
    #     # print(env.action_space.sample())
    #
    #     if terminated or truncated:
    #         observation, info = env.reset()
    #
    # print(range(env.action_space.n))
    env.close()
