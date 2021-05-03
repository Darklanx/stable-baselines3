from typing import Any, Dict, Optional, Type, Union, Tuple

import torch as th
import gym
from gym import spaces
from torch.nn import functional as F
from torch.distributions import Bernoulli, Categorical, Normal
from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.offpac.policies import OffPACPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit, Transition
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import explained_variance, polyak_update, get_linear_fn, is_vectorized_observation
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer, TrajectoryBuffer, Trajectory
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps

class OffPAC(OffPolicyAlgorithm):

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        target_update_interval: int = 10000,
        tau: float = 0.9,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        train_freq: Union[int, Tuple[int, str]] = (128, "episode"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        create_eval_env: bool = False,
        _init_setup_model: bool = True,
        KL: bool = False,
        exploration_fraction: float = 0.5,
        exploration_initial_eps: float = 0.9,
        exploration_final_eps: float = 0.05,
        support_multi_env: bool = True
    ):

        super(OffPAC, self).__init__(
            policy,
            env,
            OffPACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
            support_multi_env=support_multi_env
        )
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.KL = KL
        self.target_update_interval = target_update_interval
        self.trajectory_buffer = None

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        '''
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)
        '''
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(OffPAC, self)._setup_model()
        self._create_aliases()
        self.trajectory_buffer = TrajectoryBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device
        )
        self.replay_buffer = self.trajectory_buffer
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _store_transition(
        self, 
        buffer,
        trajectory
    ) -> None:
        buffer.add(trajectory)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
        return action, state

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])

        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
          unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        buffer: TrajectoryBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``TrajectoryBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param trajectory_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        # assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0
            trajectory = Trajectory(self.device)
            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)
                # log_probs = self.policy.forward(th.tensor(self._last_obs).to(self.device), th.tensor(action).to(self.device))
                log_probs = self.policy.get_action_log_probs(th.tensor(self._last_obs).to(self.device), th.tensor([action]).to(self.device))
                log_prob = log_probs[0].item()
                prob = th.exp(th.tensor(log_prob))
                prob = (1 - self.exploration_rate) * prob + (self.exploration_rate) * (1.0 / self.action_space.n)
                if prob > 1:
                    print("prob > 1!!! => Code in offpac.py")
                    print(prob)
                    print(th.tensor(log_prob))
                    exit()
                new_obs, reward, done, infos = env.step(buffer_action)
                # print(new_obs.size)
                # Rescale and perform action

                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1
                

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward
                
                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)
                
                trajectory.add(Transition(self._last_obs, action[0], reward[0], new_obs[0], done[0], prob))
                self._last_obs = new_obs

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break
                # if done == True and self._episode_num % 100 == 0:
                    # print("Episode reward: ", episode_reward)
                    # print("Episode timestamp: ", episode_timesteps)

            self._store_transition(buffer, trajectory)
            # print("Episode reward: ", episode_reward)
            # print("Episode timestamp: ", episode_timesteps)
            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
            
    def _on_step(self) -> None:
        """
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
        
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
    
    def padding_tensor(self, sequences, device, max_len=None):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        if max_len is None:
            max_len = max([s.size(0) for s in sequences])
        s = sequences[0]
        if s.dim() >= 2:
            list_dims = [num, max_len]
            for d in list(s.size())[1:]:
                list_dims.append(d)
            out_dims = tuple(list_dims)
        else:
            out_dims = (num, max_len)
        out_tensor = th.zeros(out_dims)
        mask = th.zeros((num, max_len))
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if s.dim() == 2:
                out_tensor[i, -length:, :] = tensor
            else:
                out_tensor[i, -length:] = tensor
            mask[i, -length:] = 1
        return out_tensor.to(device), mask.to(device)

    def train(self, gradient_steps: int, batch_size: int=100) -> None:
        self._update_learning_rate(self.policy.optimizer)
        value_losses = []
        policy_losses = []
        for _ in range(gradient_steps):
            trajectories = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            ''''''

            traj_states, traj_actions, traj_rewards, traj_dones, traj_values = [], [], [], [], []
            traj_Q_values, traj_rhos, traj_log_probs = [], [], []
            # traj_latents, traj_old_distributions = [], []
            traj_latents = []
            max_len = 0
            indexes = []
            next_state_values = []
            for traj in trajectories:
                max_len = max(max_len, len(traj))
                states, actions, rewards, next_states, dones, probs = traj.get_tensors()
                
                # KL theta
                latent, old_distribution = self.policy.get_policy_latent(states)
                latent = latent - th.mean(latent, dim=1)[0].view(-1,1)

                if states.dim() == 1:
                    states = states.unsqueeze(0)

                Q_values, log_cur_probs, _  = self.policy.evaluate_actions(states, actions)
                cur_probs = th.exp(log_cur_probs)
                values = self.policy.compute_value(states, self.q_net) # checked
                next_state_value = self.policy.compute_value(next_states, self.q_net)[-1]
                next_state_values.append(next_state_value)
                behav_probs = (1 - self.exploration_rate) * cur_probs + (self.exploration_rate) * (1.0 / self.action_space.n)
                rhos = cur_probs / behav_probs
                traj_states.append(states)
                traj_latents.append(latent)
                # traj_old_distributions.append(old_distribution)
                traj_actions.append(actions)
                traj_rewards.append(rewards)
                traj_dones.append(dones)
                traj_values.append(values)
                traj_Q_values.append(Q_values.squeeze(1))
                traj_rhos.append(rhos)
                traj_log_probs.append(log_cur_probs)
            # print(max_len)
            # print(min_len)
            # self._on_step()
            # self._on_step()
            traj_states, masks = self.padding_tensor(traj_states, self.device, max_len)
            traj_actions, _ = self.padding_tensor(traj_actions, self.device, max_len)
            traj_rewards, _ = self.padding_tensor(traj_rewards, self.device, max_len)
            traj_dones, _ = self.padding_tensor(traj_dones, self.device, max_len)
            traj_values, _ = self.padding_tensor(traj_values, self.device, max_len)
            traj_Q_values, _ = self.padding_tensor(traj_Q_values, self.device, max_len)
            traj_rhos, _ = self.padding_tensor(traj_rhos, self.device, max_len)
            traj_log_probs, _ = self.padding_tensor(traj_log_probs, self.device, max_len)
            traj_latents, _ = self.padding_tensor(traj_latents, self.device, max_len)
            # print(traj_latents.size())
            traj_old_latents = traj_latents.clone()
            # traj_old_distributions, _ = self.padding_tensor(traj_old_distributions, self.device)

            num = traj_dones.size(0)
            Q_rets = th.zeros((num, max_len), dtype=th.float).to(self.device)
            advantages = th.zeros((num, max_len), dtype=th.float).to(self.device)
            next_state_values = th.tensor(next_state_values).to(self.device)
            alpha = th.zeros((num, max_len), dtype=th.float).to(self.device)
            with th.no_grad():
                dones = traj_dones[:, -1]
                Q_rets[:, -1] = traj_rewards[:, -1] + self.gamma * (1-dones) * next_state_values
                
                advantages[:, -1] =  Q_rets[:, -1] - traj_values[:, -1]
                for i in reversed(range(max_len-1)):
                    Q_rets[:, i] = traj_rewards[:, i] + self.gamma * (th.clamp(traj_rhos[:, i+1], max=1) * (Q_rets[:, i+1] - traj_Q_values[:, i+1]) + traj_values[:, i+1]) 
                    advantages[:, i] = Q_rets[:, i] - traj_values[:, i]
                Q_rets = Q_rets * masks
            value_loss = F.mse_loss(th.flatten(traj_Q_values), th.flatten(Q_rets), reduction='mean')
            # value_loss = F.smooth_l1_loss(th.flatten(traj_Q_values), th.flatten(Q_rets), reduction='mean')

            # print("value_loss: ", F.mse_loss(th.flatten(traj_Q_values), th.flatten(Q_rets), reduction='sum'))
            # print(traj_log_probs.size())
            if not self.KL:
                # print(traj_rhos[0])
                # print(advantages[0][0:5])
                
                policy_loss = -(traj_rhos.detach() * advantages.detach() * traj_log_probs * masks).mean()
            else:
                with th.no_grad():
                    traj_action_probs = th.exp(traj_log_probs)
                    alpha = 1.0 / traj_action_probs 
                    alpha = th.clamp(alpha, max=30)
                    # _j = th.arange(latent.size(0)).long()
                    # print("adv: ", advantages)
                    addition = (th.sign(advantages) * alpha * (1-traj_action_probs)).unsqueeze(-1)
                    # addition = th.ones_like(th.sign(advantages) * alpha * (1-traj_action_probs)).unsqueeze(-1)
                    # print(addition)
                    assert addition.size()  == traj_latents.gather(2, traj_actions.long()).size()
                    # latent_correspond = traj_latents.gather(2, traj_actions.long())
                    # print(addition.size())
                    # print(traj_latents.size())
                    # print(traj_latents.sum())
                    # print("actions: ", traj_actions)
                    # print(traj_latents.size)
                    traj_latents = traj_latents + th.zeros_like(traj_latents).scatter_(2, traj_actions.long(), addition)
                    # print(traj_latents)
                    # print(traj_latents.sum())
                    traj_latents = traj_latents.clamp(min=-100, max=100)



                old_distribution = Categorical(probs=F.softmax(traj_old_latents.view(-1, self.action_space.n), dim=1))
                    
                new_distribution = Categorical(probs=F.softmax(traj_latents.view(-1, self.action_space.n).detach(), dim=1))
                policy_loss = th.distributions.kl_divergence(old_distribution, new_distribution).mean()

                # policy_loss = 0.0
                # for i in range(num):
                    
                #     # print('t: ', traj_latents[i].size())
                #     old_distribution = Categorical(probs=F.softmax(traj_old_latents[i], dim=1))
                    
                #     new_distribution = Categorical(probs=F.softmax(traj_latents[i].detach(), dim=1))
                #     policy_loss += th.distributions.kl_divergence(old_distribution, new_distribution).sum()
                
                # policy_loss /= num

            
            # print("policy_loss: ", policy_loss)       
            '''
            value_loss = th.tensor(0.0).to(self.device)
            policy_loss = th.tensor(0.0, dtype=th.float64).to(self.device)
            for traj in trajectories:
                with th.no_grad():
                    target_q_values, advantages, rhos = [], [], []
                    states, actions, rewards, next_states, dones, probs = traj.get_tensors()
                    if states.dim() == 1:
                        states = states.unsqueeze(0)
                    values = self.policy.compute_value(states, self.q_net) # checked
                    next_state_values = self.policy.compute_value(next_states, self.q_net) # checked
                    if dones[-1]:
                        Q_ret = th.tensor([0]).to(self.device)
                    else:
                        Q_ret = th.tensor([next_state_values[-1].item()]).to(self.device)
                    # behav_probs = probs

                    Q_values, log_cur_probs, _  = self.policy.evaluate_actions(states, actions)

                    cur_probs = th.exp(log_cur_probs).view(-1, 1)
                    behav_probs = (1 - self.exploration_rate) * cur_probs + (self.exploration_rate) * (1.0 / self.action_space.n)
                    for i in reversed(range(len(traj))):
                        Q_ret = rewards[i] + self.gamma * Q_ret
                        
                        # print("probs: ", cur_probs[i], behav_probs[i])
                        # try:
                        #     assert(abs(cur_probs[i].item() - behav_probs[i].item()) < 1e-5)
                        # except AssertionError as e:
                        #     print(abs(cur_probs[i].item() - behav_probs[i].item()))
                        #     print(cur_probs[i])
                        #     print(behav_probs[i])
                        #     exit()
                        rho = cur_probs[i] / behav_probs[i]
                        rhos.insert(0, rho)
                        # rho = th.min(th.tensor([1, rho])).detach()
                        rho = th.clamp(rho, max=1)
                        target_q_values.insert(0, Q_ret.detach())
                        advantages.insert(0, Q_ret.detach() - values[i])
                        Q_ret = rho * (Q_ret - Q_values[i].detach()) + values[i].detach()
                        

                current_q_values, log_probs, _  = self.policy.evaluate_actions(states, actions)
                target_q_values = th.cat(target_q_values).to(self.device).view(-1, 1)
                assert(current_q_values.size() == target_q_values.size())
                
                advantages = th.cat(advantages).to(self.device).detach()
                rhos = th.tensor(rhos).to(self.device).detach()
                value_loss += F.mse_loss(current_q_values, target_q_values, reduction='sum')
                if not self.KL:
                    policy_loss += -(rhos * advantages.detach() * log_probs).sum().double()
                else:
                    latent, old_distribution = self.policy.get_policy_latent(states)
                    latent = latent.detach()
                    with th.no_grad():
                        latent = latent - th.mean(latent, dim=1)[0].view(-1,1)
                        
                        action_prob = th.exp(log_probs) 
                        alpha  = 1.0 / action_prob
                        alpha = th.clamp(alpha, 0, 30)
                        _j = th.arange(latent.size(0)).long()
                        action_idx = actions.squeeze()
                        # print(th.sign(advantages.detach()) * alpha * (1-action_prob).squeeze())
                        # print(action_idx[0:5])
                        # print(latent[:, action_idx][0:5])
                        # print(latent[_j, action_idx][0:5])
                        # print('test:')
                        # print((th.sign(advantages.detach()) * alpha * (1-action_prob).squeeze()).size())
                        # print(latent)
                        latent[_j, action_idx] += (th.sign(advantages.detach()) * alpha * (1-action_prob)).squeeze()
                        # print(latent)
                        latent = th.clamp(latent, -100, 100)
                        new_distribution = Categorical(probs=F.softmax(latent, dim=1))
                        # print(old_distribution.probs)
                        # print(new_distribution.probs)

                    policy_loss += th.distributions.kl_divergence(old_distribution, new_distribution).sum()
            '''
            
            '''
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                next_values = self.policy.compute_value(replay_data.next_observations, self.q_net_target).view(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_values
            
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values, log_prob, entropy  = self.policy.evaluate_actions(replay_data.observations, replay_data.actions)
            
            # A(s,a) = Q(s,a) - V(s)
            current_values = self.policy.compute_value(replay_data.observations, self.q_net).view(-1, 1)
            advantages = current_q_values - current_values
            '''

            value_losses.append((self.vf_coef * value_loss).item())

            '''
            if not self.KL:
                policy_loss = -(advantages.detach() * log_prob).mean()
            else:
                latent, old_distribution = self.policy.get_policy_latent(replay_data.observations)
                latent = latent.detach()
                with th.no_grad():
                    latent = latent - th.mean(latent, dim=1)[0].view(-1,1)
                    
                    action_prob = th.exp(log_prob)
                    alpha  = 1.0 / action_prob / 10
                    alpha = th.clamp(alpha, -1, 1)
                    _j = th.arange(latent.size(0)).long()
                    action_idx = replay_data.actions.squeeze()
                    latent[_j, action_idx] += (th.sign(advantages.detach()) * alpha * (1-action_prob)).squeeze()
                    

                    latent = th.clamp(latent, -50, 50)
                    # print(latent)

                    new_distribution = Categorical(probs=F.softmax(latent, dim=1))
                    # print(new_distribution.batch_shape)
                    # exit()

                # print(type(old_distribution))
                # print(type(new_distribution))
                policy_loss = th.distributions.kl_divergence(old_distribution, new_distribution).mean()
                if th.isinf(th.sum(policy_loss)).item():
                    print("INFF")
                    print("latent: ", latent)
                    print("alpha:  ", alpha)
                    exit()
            '''
            policy_losses.append(policy_loss.item())

            # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            loss = policy_loss + self.vf_coef * value_loss
            # loss=policy_loss
            
            
            # Optimization step
            # print("value_loss 2: ",value_loss)
            # print("policy_loss 2: ", policy_loss / num)
            self.policy.optimizer.zero_grad()
            loss.backward()
            # print(self.policy.action_net)
            # print(self.policy.action_net.weight.grad)
            
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            '''
            /////
            '''

        self._n_updates += 1

        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/policy_loss", np.mean(policy_losses))
        logger.record("rollout/epsilon", self.exploration_rate)
        
    '''
    def train2(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # TODO: avoid second computation of everything because of the gradient
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/explained_variance", explained_var)
        logger.record("train/entropy_loss", entropy_loss.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())
    '''
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OFFPAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OFFPAC":

        return super(OffPAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            use_trajectory_buffer=True
        )
