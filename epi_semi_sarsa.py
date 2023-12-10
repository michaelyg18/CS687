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
            nn.Linear(input_dim, input_dim*5),
            nn.LeakyReLU(),
            nn.Linear(input_dim*5, input_dim*2),
            # nn.ReLU(),
            # nn.Linear((input_dim + 1) * 2, (input_dim + 1)),
            nn.Tanh(),
            nn.Linear(input_dim*2, 1)
        )

    def forward(self, x):
        return self.mlp(x)

# q_func is a neural network that returns a prediction
def episodic_semigradient_nstep_sarsa(env, q_func, alpha, gamma, epsilon, n, num_episodes):
    # initialize value-function weights arbitrarily
    loss_func = nn.MSELoss()  # does reduction matter?
    optimizer = torch.optim.Adam(q_func.parameters(), lr=alpha)
    return_hist = []
    # all store and access operations (S, A, R) can take index mod n+1?

    # loop for each episode
    for episode in range(1, num_episodes + 1):
        # initialize and store non-terminal S0
        state, _ = env.reset()
        states = [state]
        # states = [0] * (n + 1)

        # select and store action A0 ~ pi(·|S0), or epsilon greedy wrt q(S0, ·, w)
        action = epsilon_greedy(env, state, q_func, epsilon/np.sqrt(episode))
        # action = epsilon_greedy(env, state, q_func, epsilon)
        # action = softmax(env, state, q_func)
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
                    next_action = epsilon_greedy(env, states[t], q_func, epsilon/np.sqrt(episode))
                    # next_action = epsilon_greedy(env, states[t], q_func, epsilon)
                    # next_action = softmax(env, next_state, q_func)
                    actions.append(next_action)

                # print("Step:", t, "State:", states[t], "action:", actions[t])

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
                    # q_pred = q_func(state_action_tau_n).detach()
                    # print(q_pred.item())
                    # G += q_func(state_action_tau_n)
                    G += gamma**n * q_pred

                # weight update
                state_action_tau = torch.tensor(np.append(states[tau], actions[tau]))
                q_pred = q_func(state_action_tau)
                # MSE of St and Ut
                # print(G)
                loss = loss_func(q_pred, G)
                # optimizer.zero_grad()
                # print(loss.item(), q_pred.item(), G.item())  # getting stuck?
                loss.backward()
                optimizer.step()

            if tau == T_cap - 1:
                break

            # next timestep
            t += 1

        total_return = np.sum(rewards)
        return_hist.append(total_return)
        print("Episode", episode, "Total return:", total_return)
        # print("Model weights:", list(q_func.parameters()))

    return return_hist


def epsilon_greedy(env, state, q_func, epsilon):
    # pick greedy with 1-epsilon chance
    if random.random() > epsilon:
        # decide action with softmax loss?
        # for every possible action in current state, get q value and take argmax
        q_vals = []
        for action in range(env.action_space.n):
            q_pred = q_func(torch.tensor(np.append(state, action)))
            # q_pred = q_func(torch.tensor(np.append(state, action))).detach().numpy()
            q_vals.append(q_pred)

        return torch.argmax(torch.tensor(q_vals)).item()
        # return np.argmax(q_vals)

    # explore with epsilon chance
    else:
        # random action in action space (n choices in discrete case)
        return env.action_space.sample()

def softmax(env, state, q_func):
    q_vals = []
    actions = range(env.action_space.n)
    for action in actions:
        q_pred = q_func(torch.tensor(np.append(state, action))).detach().numpy()
        q_vals.append(q_pred)

    # softmax
    softmax_probs = (np.exp(q_vals)/sum(np.exp(q_vals))).flatten()
    return np.random.choice(actions, p=softmax_probs)


if __name__ == '__main__':

    # mountain car MDP
    # see whats going on with human render mode, but runs a lot slower
    # set max steps higher for mountaincar
    # env = gym.make('MountainCar-v0', render_mode=None)
    # env = gym.make('MountainCar-v0', render_mode="human")
    # env._max_episode_steps = 1000
    env = gym.make('LunarLander-v2', render_mode=None)
    # env = gym.make('LunarLander-v2', render_mode="human")
    observation, info = env.reset(seed=42)  # set RNG generation seed for future episodes

    # initialization
    input_dim = observation.shape[0] + 1  # input dimension to approximator network is state space + the action taken
    # maybe add input dim of state * action to relate them more?
    q_func = ActionValueApproximator(input_dim)
    q_func.double()
    # q_func.int()
    alpha = 1e-3
    gamma = .9  # can't be 1 for lunarlander
    epsilon = 1
    n = 4
    num_episodes = 500

    return_hist = episodic_semigradient_nstep_sarsa(env, q_func, alpha, gamma, epsilon, n, num_episodes)
    env.close()

    plt.figure()
    plt.plot(return_hist)
    plt.title("Total return over episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Total return")
    plt.show()

