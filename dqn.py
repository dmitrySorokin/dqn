from agent import DQNAgent
import utils
from replay_buffer import ReplayBuffer
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0]
        new_s, r, done, _ = env.step(action)
        exp_replay.add(s, action, r, new_s, done)
        s = new_s
        if done:
            s = env.reset()
        sum_rewards += r

    return sum_rewards, s


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network,
                    device,
                    gamma=0.99,
                    check_shapes=False):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float)  # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)  # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]

    # compute V*(next_states) using predicted next q-values
    next_state_values, _ = predicted_next_qvalues.max(-1)

    assert next_state_values.dim(
    ) == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * next_state_values * is_not_done

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"

    return loss


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    solved_games = 0
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for i in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                if i < 49:
                  solved_games += 1
                break

        rewards.append(reward)
    return np.mean(rewards), solved_games / n_games


class DQN(object):
    def __init__(self,
                 env_creator,
                 device,
                 buffer_size,
                 save_dir,
                 timesteps_per_epoch=1,
                 batch_size=32,
                 total_steps=5 * 10 ** 5,
                 decay_rate=0.1,
                 init_epsilon=1,
                 final_epsilon=0.02,
                 loss_freq=50,
                 refresh_target_network_freq=500,
                 eval_freq=500,
                 max_grad_norm=50):

        self.env_creator = env_creator
        self.env = env_creator()
        n_actions = self.env.action_space.n
        state_shape = self.env.observation_space.shape

        self.save_dir = save_dir
        self.buffer_size = buffer_size
        self.timesteps_per_epoch = timesteps_per_epoch
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.decay_steps = decay_rate * total_steps
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.loss_freq = loss_freq
        self.refresh_target_network_freq = refresh_target_network_freq
        self.eval_freq = eval_freq
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.writer = SummaryWriter('runs')

        self.agent = DQNAgent(state_shape, n_actions, epsilon=0.5).to(device)
        self.target_network = DQNAgent(state_shape, n_actions, epsilon=0.5).to(device)
        self.target_network.load_state_dict(self.agent.state_dict())

    def save(self):
        torch.save(self.agent.state_dict(), self.save_dir)

    def load(self):
        state = torch.load(self.save_dir)
        self.agent.load_state_dict(state)

    def learn(self):
        exp_replay = self._fill_replay_buffer()
        state = self.env.reset()
        opt = torch.optim.Adam(self.agent.parameters(), lr=1e-5)

        for step in range(self.total_steps):
            if not utils.is_enough_ram():
                print('less that 100 Mb RAM available, freezing')
                print('make sure everythin is ok and make KeyboardInterrupt to continue')
                try:
                    while True:
                        pass
                except KeyboardInterrupt:
                    pass

            self.agent.epsilon = utils.linear_decay(self.init_epsilon, self.final_epsilon, step, self.decay_steps)

            # play
            _, state = play_and_record(state, self.agent, self.env, exp_replay, self.timesteps_per_epoch)

            # train
            obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(self.batch_size)

            # loss
            loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch,
                                   self.agent, self.target_network, self.device, gamma=0.99, check_shapes=False)

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            opt.step()
            opt.zero_grad()

            if step % self.refresh_target_network_freq == 0:
                self.target_network.load_state_dict(self.agent.state_dict())

            if step % self.loss_freq == 0:
                td_loss = loss.data.cpu().item()
                self.writer.add_scalar('Train/Loss', td_loss, step)
                self.writer.add_scalar('Train/GradNorm', grad_norm, step)

            if step % self.eval_freq == 0:
                mean_reward, solved_games = evaluate(self.env_creator(seed=step), self.agent, n_games=15, greedy=True)
                self.writer.add_scalar('Train/MeanReward', mean_reward, step)
                self.writer.add_scalar('Train/SolvedGames', solved_games, step)

                initial_state_q_values = self.agent.get_qvalues([self.env_creator(seed=step).reset()])
                self.writer.add_scalar('Train/QValues', np.max(initial_state_q_values), step)

    def _fill_replay_buffer(self):
        state = self.env.reset()
        exp_replay = ReplayBuffer(self.buffer_size)
        for _ in range(1000):
            if not utils.is_enough_ram(min_available_gb=0.1):
                print("""
                    Less than 100 Mb RAM available. 
                    Make sure the buffer size in not too huge.
                    Also check, maybe other processes consume RAM heavily.
                    """
                      )
                break
            play_and_record(state, self.agent, self.env, exp_replay, n_steps=10 ** 2)
            if len(exp_replay) == self.buffer_size:
                break
        print(len(exp_replay))
        return exp_replay
