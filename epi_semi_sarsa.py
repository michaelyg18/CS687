import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
import torch
import torch.nn as nn


# neural network to approximate action-value
class ActionValueApproximator(nn.Module):
    # input is state space, output is action space
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # nn.LayerNorm(256),
            # nn.Linear(256, 64),
            # nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = torch.tensor(x).double()
        return self.mlp(x)

def episodic_semigradient_nstep_sarsa(env, q_func, optimizer, gamma, epsilon, n, num_episodes):
    # initialize value-function weights arbitrarily
    # all store and access operations (S, A, R) can take index mod n+1
    loss_func = nn.MSELoss()
    return_hist = []
    step_hist = []

    # loop for each episode
    for episode in range(1, num_episodes + 1):
        # initialize and store non-terminal S0
        state, _ = env.reset()
        states = [state]

        # select and store action A0 ~ pi(·|S0), or epsilon greedy wrt q(S0, ·, w)
        # action = epsilon_greedy(env, state, q_func, epsilon/np.sqrt(episode))
        action = softmax(env, state, q_func)
        actions = [action]
        rewards = [0]

        # loop for t = 0, 1, 2...
        t = 0
        T_cap = np.inf
        while True:
            if t < T_cap:
                # take action At
                # observe and store next reward as Rt+1, and the next state as St+1
                next_state, reward, terminated, truncated, info = env.step(actions[t])
                states.append(next_state)
                rewards.append(reward)

                # if St+1 is terminal, then:
                if terminated or truncated:
                    T_cap = t + 1
                else:
                    # select and store At+1 ~ pi(·|St+1) or epsilon greedy wrt q(St+1, ·, w)
                    # next_action = epsilon_greedy(env, states[t], q_func, epsilon/np.sqrt(episode))
                    next_action = softmax(env, next_state, q_func)
                    actions.append(next_action)

            # tau is time whose estimate is being updated
            tau = t - n + 1
            if tau >= 0:
                i = tau + 1
                sum_lim = min(tau + n, T_cap)
                # summation formula: take slice of reward list, multiply in powers of gamma
                G = np.sum(np.multiply(rewards[i:sum_lim + 1], np.power(gamma, np.arange(sum_lim - i + 1))))
                G = torch.tensor(G).double()

                if tau + n < T_cap:
                    # pass state and action at time tau+n to q_func, get a prediction back (forward pass)
                    q_pred = q_func(states[tau+n])[actions[tau+n]].detach()
                    G += gamma**n * q_pred

                # weight update step
                # MSE of St and G, then backprop
                q_pred = q_func(states[tau])[actions[tau]]
                loss = loss_func(q_pred, G)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if tau == T_cap - 1:
                break

            # next timestep
            t += 1

        total_return = np.sum(rewards)
        return_hist.append(total_return)
        step_hist.append(t)
        print("Episode", episode, "Total return:", total_return)

    return return_hist, step_hist


def epsilon_greedy(env, state, q_func, epsilon):
    # pick greedy with 1-epsilon chance
    if random.random() > epsilon:
        # for every possible action in current state, get q value and take argmax
        q_vals = q_func(state).detach().numpy()
        return np.argmax(q_vals)

    # explore with epsilon chance
    else:
        # random action in action space (n choices in discrete case)
        return env.action_space.sample()

def softmax(env, state, q_func):
    # create tensor from states, turn it to double, detach it from network and convert to numpy
    q_vals = q_func(state).detach().numpy()
    # numerically stabilize by subtracting max action value from vector, then softmax
    q_vals = q_vals - np.max(q_vals)
    softmax_probs = (np.exp(q_vals)/sum(np.exp(q_vals))).flatten()

    return np.random.choice(env.action_space.n, p=softmax_probs)


if __name__ == '__main__':
    # create MDP
    # see episodes with human render mode, but runs a lot slower

    # env = gym.make('MountainCar-v0', render_mode=None)
    # env = gym.make('MountainCar-v0', render_mode="human")
    # env._max_episode_steps = 2500  # set max steps higher for mountaincar

    env = gym.make('LunarLander-v2', render_mode=None)
    # env = gym.make('LunarLander-v2', render_mode="human")

    # env = gym.make('CartPole-v1', render_mode=None)
    # env = gym.make('CartPole-v1', render_mode="human")
    observation, info = env.reset(seed=5)  # set RNG generation seed for future episodes

    # initialization
    input_dim = observation.shape[0]  # input dimension to approximator network is state space
    output_dim = env.action_space.n  # output dimension is action space

    # maybe add input dim of state * action to relate them more?
    q_func = ActionValueApproximator(input_dim, output_dim).double()
    # q_func.double()
    alpha = 1e-2
    gamma = 0.9
    epsilon = 1
    n = 4
    num_episodes = 10

    # initialize optimizer and run algorithm
    optimizer = torch.optim.Adam(q_func.parameters(), lr=alpha)
    return_hist = episodic_semigradient_nstep_sarsa(env, q_func, optimizer, gamma, epsilon, n, num_episodes)
    env.close()

    plt.figure()
    plt.plot(return_hist)
    plt.title("Total return over episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Total return")
    plt.show()

