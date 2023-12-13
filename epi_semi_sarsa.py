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
    total_steps = 0
    total_step_hist = []
    loss_hist = []

    # loop for each episode
    for episode in range(1, num_episodes + 1):
        total_loss = 0
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
                    q_pred = q_func(states[tau + n])[actions[tau + n]].detach()
                    G += gamma ** n * q_pred

                # weight update step
                # MSE of St and G, then backprop
                q_pred = q_func(states[tau])[actions[tau]]
                loss = loss_func(q_pred, G)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if tau == T_cap - 1:
                break

            # next timestep
            t += 1

        total_return = np.sum(rewards)
        return_hist.append(total_return)
        total_steps += t
        total_step_hist.append(total_steps)
        loss_hist.append(total_loss)
        if episode % 100 == 0:
            print("Episode", episode, "Total return:", total_return)

    return return_hist, total_step_hist, loss_hist


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
    softmax_probs = (np.exp(q_vals) / sum(np.exp(q_vals))).flatten()

    return np.random.choice(env.action_space.n, p=softmax_probs)


def run_stats(env, alpha, gamma, epsilon, n, num_episodes, num_trials, env_name):
    observation, info = env.reset(seed=5)  # set RNG generation seed for future episodes
    # initialization
    input_dim = observation.shape[0]  # input dimension to approximator network is state space
    output_dim = env.action_space.n  # output dimension is action space

    # maybe add input dim of state * action to relate them more?
    q_func = ActionValueApproximator(input_dim, output_dim).double()

    trial_hist = []
    trial_loss_hist = []

    # initialize optimizer and run algorithm
    optimizer = torch.optim.Adam(q_func.parameters(), lr=alpha)

    for i in range(num_trials):
        return_hist, step_hist, loss_hist = episodic_semigradient_nstep_sarsa(env, q_func, optimizer, gamma, epsilon, n,
                                                                              num_episodes)
        trial_hist.append(return_hist)
        trial_loss_hist.append(loss_hist)
        print(f"Run {i+1} complete.")

    # single run
    ax1 = plt.figure().gca()
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    plt.plot(trial_hist[-1])
    plt.title(f"esn-SARSA return over one run ({env_name})")
    plt.xlabel("Episodes")
    plt.ylabel("Total return")
    plt.savefig(f'{env_name} single run.png')
    # plt.show()

    # average return over multiple runs
    trial_hist = np.array(trial_hist)
    mean_hist = np.mean(trial_hist, axis=0)
    std_hist = np.std(trial_hist, axis=0)
    ax1 = plt.figure().gca()
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    plt.plot(range(1, num_episodes + 1), mean_hist, color='b')
    plt.fill_between(range(1, num_episodes + 1), mean_hist - std_hist, mean_hist + std_hist, color='c')
    plt.title(f"Average return of esn-SARSA over {num_trials} runs ({env_name})")
    plt.xlabel("No. episodes")
    plt.ylabel("Average return")
    plt.savefig(f'{env_name} avg return.png')
    # plt.show()

    # average loss over multiple runs
    trial_loss_hist = np.array(trial_loss_hist)
    mean_hist = np.mean(trial_loss_hist, axis=0)
    std_hist = np.std(trial_loss_hist, axis=0)
    ax1 = plt.figure().gca()
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    plt.plot(range(1, num_episodes + 1), mean_hist, color='b')
    plt.fill_between(range(1, num_episodes + 1), mean_hist - std_hist, mean_hist + std_hist, color='c')
    plt.title(f"Average Loss of esn-SARSA over {num_trials} runs ({env_name})")
    plt.xlabel("No. episodes")
    plt.ylabel("Average Loss")
    plt.savefig(f"{env_name} avg loss.png")
    # plt.show()

    # steps vs episodes
    ax1 = plt.figure().gca()
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    plt.plot(step_hist, range(1, num_episodes + 1))
    plt.title(f"esn-SARSA steps vs episodes ({env_name})")
    plt.xlabel("Steps")
    plt.ylabel("Episodes")
    plt.savefig(f'{env_name} steps.png')
    # plt.show()


if __name__ == '__main__':
    # create MDP
    # see episodes with human render mode, but runs a lot slower

    # env = gym.make('MountainCar-v0', render_mode=None)
    # env = gym.make('MountainCar-v0', render_mode="human")
    # env._max_episode_steps = 2500  # set max steps higher for mountaincar
    env = gym.make('LunarLander-v2', render_mode=None)
    # env = gym.make('LunarLander-v2', render_mode="human")
    run_stats(env, alpha=5e-4, gamma=0.9, epsilon=1, n=4, num_episodes=500, num_trials=5, env_name="Lunar Lander")
    env.close()

    env = gym.make('CartPole-v1', render_mode=None)
    # env = gym.make('CartPole-v1', render_mode="human")
    run_stats(env, alpha=1e-4, gamma=0.9, epsilon=1, n=4, num_episodes=10000, num_trials=5, env_name="Cart Pole")
    env.close()


