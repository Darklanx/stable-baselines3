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
from stable_baselines3.common.utils import explained_variance, polyak_update, get_linear_fn, is_vectorized_observation, get_ms
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer, TrajectoryBuffer, Trajectory, RolloutBuffer
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
import time
class OffPAC(OffPolicyAlgorithm):

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        target_update_interval: int = 10,
        behav_update_interval: int = 100,
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
        exploration_initial_eps: float = 0.5,
        exploration_final_eps: float = 0.01,
        support_multi_env: bool = True,
        share: bool = True,
        max_alpha: int = 10,
        reg_coef: float = 1.0,
        behav_tau: float = 1.0,
        use_rms_prop: bool = True,
        rms_prop_eps: float = 1e-5,
        use_v_net: bool=False
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
            support_multi_env=support_multi_env,
            share=share
        )
        self.use_v_net = use_v_net
        self.behav_tau = behav_tau
        self.reg_coef = reg_coef
        self.max_alpha = max_alpha
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.KL = KL
        self.target_update_interval = target_update_interval
        self.behav_update_interval = behav_update_interval
        self.trajectory_buffer = None
        self.n_backward = 0
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.trajectories = [Trajectory(self.device) for i in range(self.n_envs)]
        '''
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        
        '''
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)
        if _init_setup_model:
            self._setup_model()

        self.rollout_buffer = RolloutBuffer(
            self.train_freq.frequency,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )
        

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
        self.behav_net = self.policy.behav_net
        self.action_net = self.policy.action_net
        self.v_mlp_extractor = self.policy.v_mlp_extractor
        self.v_mlp_extractor_target = self.policy.v_mlp_extractor_target
        self.a_mlp_extractor = self.policy.a_mlp_extractor
        self.a_mlp_extractor_target = self.policy.a_mlp_extractor_target
        self.value_net = self.policy.value_net

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
        use_behav: bool = False
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
        action_net = self.behav_net if use_behav else self.action_net
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic, use_behav)
        return action, state

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None, use_behav:bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        # action_net = self.behav_net if use_target else self.action_net
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            if self.n_envs == 1:
                unscaled_action = np.array([self.action_space.sample()])
            else:
                unscaled_action = np.array([self.action_space.sample() for i in range(self.n_envs)])

        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
          unscaled_action, _ = self.predict(self._last_obs, deterministic=False, use_behav=use_behav)

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
        self.rollout_buffer.reset()
        


        done = np.array([False for i in range(self.n_envs)])
        episode_reward, episode_timesteps = [0.0 for i in range(self.n_envs)], [0 for i in range(self.n_envs)]
        if train_freq.unit == TrainFrequencyUnit.STEP:
            self.trajectories = [Trajectory(self.device) for i in range(self.n_envs)]
            
        while True:
            ms = [0]
            get_ms(ms)
            
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise()

            # Select action randomly or according to policy
            
            with th.no_grad():
                # action, buffer_action = self._sample_action(learning_starts, action_noise, use_behav=False)
                # log_probs = self.policy.get_action_log_probs(th.tensor(np.array(self._last_obs)).to(self.device), th.tensor(np.array([action])).T.to(self.device), use_behav=False)
                action, buffer_action = self._sample_action(learning_starts, action_noise, use_behav=True)
                log_probs = self.policy.get_action_log_probs(th.tensor(np.array(self._last_obs)).to(self.device), th.tensor(np.array([action])).T.to(self.device), use_behav=True)
                prob = th.exp(log_probs)
                prob = (1 - self.exploration_rate) * prob + (self.exploration_rate) * (1.0 / self.action_space.n)
                prob = prob.cpu().numpy()


            if (prob > 1).any():
                print("prob > 1!!! => Code in offpac.py")
                print(prob)
                print(th.tensor(log_probs))
                exit()

            new_obs, reward, done, infos = env.step(action)

            with th.no_grad():
                if self.use_v_net:
                    latent_pi, latent_vf, latent_sde = self.policy._get_latent(th.tensor(self._last_obs))
                    values = self.value_net(latent_vf).detach()
                else:
                    values = self.policy.compute_value(th.tensor(self._last_obs), use_target_v=False).detach()
                
            self.rollout_buffer.add(self._last_obs, action.reshape(-1, 1), reward, self._last_episode_starts, values, log_probs.flatten())


            self.num_timesteps += env.num_envs
            num_collected_steps += env.num_envs


            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)
            
            episode_reward += reward
            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, done)

            for i in range(len(self.trajectories)):
                # trajectories[i].add(Transition(self._last_obs[i], action[i], reward[i], new_obs[i], done[i], prob[i]))
                if done[i]:
                    if infos[i]['terminal_observation'].dtype == np.float64:
                        self.trajectories[i].add(Transition(self._last_obs[i], action[i], reward[i], infos[i]['terminal_observation'].astype(np.float32), done[i], prob[i]))
                    else:
                        self.trajectories[i].add(Transition(self._last_obs[i], action[i], reward[i], infos[i]['terminal_observation'], done[i], prob[i]))
                else:
                    self.trajectories[i].add(Transition(self._last_obs[i], action[i], reward[i], new_obs[i], done[i], prob[i]))
            self._last_obs = new_obs
            self._last_episode_starts = done

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is done as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            '''
            if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                # even if the episdoe is not finished, we store the trajectory because no more steps can be performed
                for traj_i, traj in enumerate(trajectories):
                    self._store_transition(buffer, traj)
                    total_timesteps.append(len(traj))
                    
                    trajectories[traj_i] = Trajectory(self.device)
                    
                    episode_rewards.append(episode_reward[traj_i])
                    episode_reward[traj_i] = 0.0
                break
            '''
            



            # store transition of finished episode, but if not more steps can be collected, treat any trajectory as an episode
            if done.any():
                num_collected_episodes += np.sum(done)
                self._episode_num += np.sum(done)
                if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()


            if train_freq.unit == TrainFrequencyUnit.STEP:
                ending = not should_collect_more_steps(train_freq, num_collected_steps//self.n_envs, num_collected_episodes//self.n_envs)
                # if ending, save all trajectories, otherwise only save done episode
                if ending:
                    for traj_i, traj in enumerate(self.trajectories):
                        self._store_transition(buffer, traj)
                        # total_timesteps.append(len(traj)) # is this line affecting anything????   
                        
                        self.trajectories[traj_i] = Trajectory(self.device)
                        
                        episode_rewards.append(episode_reward[traj_i])
                        episode_reward[traj_i] = 0.0
                    break
                else:
                    if done.any():
                        traj_indexes = [i for i in np.arange(len(self.trajectories))[done]]
                        for traj_i in traj_indexes:
                            self._store_transition(buffer, self.trajectories[traj_i])
                            # total_timesteps.append(len(traj)) # is this line affecting anything????   
                            self.trajectories[traj_i] = Trajectory(self.device)
                            episode_rewards.append(episode_reward[traj_i])
                            episode_reward[traj_i] = 0.0


            elif train_freq.unit == TrainFrequencyUnit.EPISODE:
                ending = not should_collect_more_steps(train_freq, num_collected_steps//self.n_envs, num_collected_episodes//self.n_envs)
                if done.any():
                    # if ending, save all trajectories even if not finished
                    # if not ending:
                    traj_indexes = [i for i in np.arange(len(self.trajectories))[done]]
                    for traj_i in traj_indexes:
                        self._store_transition(buffer, self.trajectories[traj_i])
                        # total_timesteps.append(len(traj)) # is this line affecting anything???? 
                        
                        self.trajectories[traj_i] = Trajectory(self.device)
                        
                        episode_rewards.append(episode_reward[traj_i])
                        episode_reward[traj_i] = 0.0
                    '''
                    else:
                        _trajectories = trajectories
                    for traj_i, traj in enumerate(_trajectories):
                        self._store_transition(buffer, traj)
                        total_timesteps.append(len(traj)) # is this line affecting anything????   
                        
                        self.trajectories[traj_i] = Trajectory(self.device)
                        
                        episode_rewards.append(episode_reward[traj_i])
                        episode_reward[traj_i] = 0.0
                    '''
                if ending:
                    break
            else:
                print(train_freq.unit)
                raise Exception("Weird train_freq.unit...")
                exit(-1)
            
        
        if done.any():
            if action_noise is not None:
                action_noise.reset()

        with th.no_grad():
            obs_tensor = th.as_tensor(new_obs).squeeze(1).to(self.device)
            if self.use_v_net:
                latent_pi, latent_vf, latent_sde = self.policy._get_latent(obs_tensor)
                values = self.value_net(latent_vf).detach()
            else:
                values = self.policy.compute_value(obs_tensor, use_target_v=False)

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)
        
        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
            
    def _on_step(self) -> None:
        """
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        pass
        # if self.num_timesteps % self.target_update_interval == 0:

        # self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)

    def _on_update(self) -> None:
        if self._n_updates % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            if not self.share:
                polyak_update(self.v_mlp_extractor.parameters(), self.v_mlp_extractor_target.parameters(), self.tau)
        
        if self._n_updates % self.behav_update_interval == 0:
            polyak_update(self.action_net.parameters(), self.behav_net.parameters(), tau=self.behav_tau)
            if not self.share:
                polyak_update(self.a_mlp_extractor.parameters(), self.a_mlp_extractor_target.parameters(), tau=self.behav_tau)
            self.trajectories = [Trajectory(self.device) for i in range(self.n_envs)]
            self.trajectory_buffer.reset()
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

        gradient_steps = max(1, min(gradient_steps, self.replay_buffer.size() // batch_size // 2))
        # print(self.replay_buffer.size())
        # print(self.replay_buffer.size())
        # print(self.replay_buffer.size() //batch_size)
        # print("steps:" ,gradient_steps)

        ms=[0]
        get_ms(ms)
        for i_gradient_step in range(gradient_steps):
            
            
            trajectories = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            trajectories.extend(self.replay_buffer.get_last(self.n_envs))
            # print(len(trajectories))
            # The following "all_{}" is for speed up by doing batched .to(device)

            all_states, all_actions, all_rewards, all_next_states, all_dones, lengths, all_probs = [], [], [], [],[], [], []
            all_next_states2 = []
            # we merge all the trajectories together for batch ".to(device)", later we extract the trajectories by using "lengths:list"
            for i, traj in enumerate(trajectories):
                states, actions, rewards, next_states, dones, probs = traj.get_tensors(device='cpu')
                lengths.append(actions.size(0))
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_next_states.append(next_states)
                all_next_states2.append(next_states[-1].unsqueeze(0))
                all_dones.append(dones)
                all_probs.append(probs)

            
            all_states = th.cat(all_states).to(self.device)
            all_actions = th.cat(all_actions).to(self.device)
            all_rewards = th.cat(all_rewards).to(self.device)
            all_next_states = th.cat(all_next_states).to(self.device)
            all_next_states2 = th.cat(all_next_states2).to(self.device)
            all_dones = th.cat(all_dones).to(self.device)
            all_probs = th.cat(all_probs).to(self.device)
            all_Q_values, all_log_cur_probs, _  = self.policy.evaluate_actions(all_states, all_actions, use_target_v=False, use_behav=False)
            with th.no_grad():
                all_target_Q_values, _, _  = self.policy.evaluate_actions(all_states, all_actions, use_target_v=True, use_behav=True)
        

            all_next_values = self.policy.compute_value(all_states, use_target_v=True, use_behav=True)
            all_next_values_last = self.policy.compute_value(all_next_states2, use_target_v=True, use_behav=True)
            traj_index_start = 0


            traj_states, traj_actions, traj_rewards, traj_dones, traj_values = [], [], [], [], []
            traj_Q_values, traj_target_Q_values, traj_rhos, traj_log_probs = [], [], [], []
            traj_latents = []
            max_len = 0
            indexes = []
            next_state_values = []
            # print('1:', ms[0] - get_ms(ms))
            for traj_i, traj in enumerate(trajectories):
            

                # t = [0]
                # get_ms(t)
                # ms = [0]
                # get_ms(ms)

                max_len = max(max_len, len(traj))
                # states, actions, rewards, next_states, dones, probs = traj.get_tensors()
                # _states, _actions, _rewards, _next_states, _dones = traj.get_tensors(device=None)
                # states, actions, rewards, next_states, dones = traj.get_tensors(device=None)

                traj_length = lengths[traj_i]
                traj_index_end = traj_index_start + traj_length
                states, actions, rewards, next_states, dones = all_states[traj_index_start:traj_index_end], all_actions[traj_index_start:traj_index_end], all_rewards[traj_index_start:traj_index_end],  all_next_states[traj_index_start:traj_index_end], all_dones[traj_index_start:traj_index_end]
                Q_values = all_Q_values[traj_index_start:traj_index_end]
                target_Q_values = all_target_Q_values[traj_index_start:traj_index_end]
                probs = all_probs[traj_index_start:traj_index_end]
                log_cur_probs = all_log_cur_probs[traj_index_start:traj_index_end]
                
                values = th.cat([all_next_values[traj_index_start:traj_index_end], all_next_values_last[traj_i].unsqueeze(0)])
                traj_index_start += traj_length
                
                
                '''
                assert _states.size(0) == states.size(0) and _actions.size(0) == actions.size(0) and _rewards.size(0) == rewards.size(0) and _next_states.size(0) == next_states.size(0) and _dones.size(0) == dones.size(0)
                '''

                # print("0:")
                # print(ms[0] - get_ms(ms))
                '''
                print("s:", states.size())
                print("a:", actions.size())
                print("r:", rewards.size())
                print("ns:", next_states.size())
                print("shapes:", dones.size())
                print("d:", probs.size())
                '''
                # KL theta
                
                latent, old_distribution = self.policy.get_policy_latent(states, use_behav=False)
                latent = latent - th.mean(latent, dim=1).view(-1,1)
                # print("1:")
                # print(ms[0] - get_ms(ms))
                if states.dim() == 1:
                    states = states.unsqueeze(0)

                # Q_values, log_cur_probs, _  = self.policy.evaluate_actions(states, actions, use_target_v=False, use_behav=False)
                '''
                with th.no_grad():
                    target_Q_values, _, _  = self.policy.evaluate_actions(states, actions, use_target_v=True, use_behav=True)
                '''
                # print("2:")
                # print(ms[0] - get_ms(ms))
                cur_probs = th.exp(log_cur_probs)
                # compute values of states (and addition last state)
                # values = self.policy.compute_value(th.cat([states, next_states[-1].unsqueeze(0)]), use_target_v=True, use_behav=True) # checked

                next_state_value = values[-1]
                values = values[:-1]

                
                # print("b: ", ms[0] - get_ms(ms))
                next_state_values.append(next_state_value)
                # behav_probs = (1 - self.exploration_rate) * cur_probs + (self.exploration_rate) * (1.0 / self.action_space.n)
                behav_probs = probs.squeeze(1)
                rhos = cur_probs / behav_probs

                traj_states.append(states)
                traj_latents.append(latent)
                traj_actions.append(actions)
                traj_rewards.append(rewards)
                traj_dones.append(dones)
                traj_values.append(values)
                traj_Q_values.append(Q_values.squeeze(1))

                traj_target_Q_values.append(target_Q_values.squeeze(1).detach())
                traj_rhos.append(rhos)
                traj_log_probs.append(log_cur_probs)
                # print("4:")
                # print(ms[0] - get_ms(ms))
                # print(t[0] - get_ms(t))

            # print(ms[0] - get_ms(ms))
                # exit()
            # print(max_len)
            # print(min_len)
            traj_states, masks = self.padding_tensor(traj_states, self.device, max_len)
            traj_actions, _ = self.padding_tensor(traj_actions, self.device, max_len)
            traj_rewards, _ = self.padding_tensor(traj_rewards, self.device, max_len)
            traj_dones, _ = self.padding_tensor(traj_dones, self.device, max_len)
            traj_values, _ = self.padding_tensor(traj_values, self.device, max_len)
            traj_Q_values, _ = self.padding_tensor(traj_Q_values, self.device, max_len)
            traj_target_Q_values, _ = self.padding_tensor(traj_target_Q_values, self.device, max_len)
            traj_rhos, _ = self.padding_tensor(traj_rhos, self.device, max_len)
            traj_log_probs, _ = self.padding_tensor(traj_log_probs, self.device, max_len)
            traj_latents, _ = self.padding_tensor(traj_latents, self.device, max_len)
            # print(traj_latents.size())
            # print(traj_actions.size())
            # exit()
            traj_old_latents = traj_latents.clone()
            # traj_old_distributions, _ = self.padding_tensor(traj_old_distributions, self.device)

            num = traj_dones.size(0)
            Q_rets = th.zeros((num, max_len), dtype=th.float).to(self.device)
            advantages = th.zeros((num, max_len), dtype=th.float).to(self.device)
            next_state_values = th.tensor(next_state_values).to(self.device)
            alpha = th.zeros((num, max_len), dtype=th.float).to(self.device)
            # if self.n_backward % 10 == 0:
                # print(th.max(traj_rhos))
            with th.no_grad():
                dones = traj_dones[:, -1]
                Q_rets[:, -1] = traj_rewards[:, -1] + self.gamma * (1-dones) * next_state_values
                
                advantages[:, -1] =  Q_rets[:, -1] - traj_values[:, -1]
                for i in reversed(range(max_len-1)):
                    Q_rets[:, i] = traj_rewards[:, i] + self.gamma * (th.clamp(traj_rhos[:, i+1], max=1) * (Q_rets[:, i+1] - traj_target_Q_values[:, i+1]) + traj_values[:, i+1]) 
                    advantages[:, i] = Q_rets[:, i] - traj_values[:, i]
                Q_rets = Q_rets * masks
            # print("2: ", ms[0] - get_ms(ms))

            value_loss = F.mse_loss(th.flatten(traj_Q_values), th.flatten(Q_rets), reduction='mean')
            # value_loss = F.smooth_l1_loss(th.flatten(traj_Q_values), th.flatten(Q_rets), reduction='mean')

            # print("value_loss: ", F.mse_loss(th.flatten(traj_Q_values), th.flatten(Q_rets), reduction='sum'))

            if not self.KL:
                policy_loss = -(traj_rhos.detach() * advantages.detach() * traj_log_probs * masks).mean()
            else:
                with th.no_grad():
                    traj_action_probs = th.exp(traj_log_probs)
                    alpha = 1.0 / traj_action_probs / 2
                    alpha = th.clamp(alpha, max=self.max_alpha)
                    # print(self.max_alpha)
                    if i_gradient_step == 0 and False:
                        print("max alpha: {}".format(th.max(alpha)))

                    # th.set_printoptions(precision=6)
                    addition = (th.sign(advantages) * (alpha * (1-traj_action_probs) + 1)).unsqueeze(-1)
                    assert addition.size()  == traj_latents.gather(2, traj_actions.long()).size()
 
                    traj_latents = traj_latents + th.zeros_like(traj_latents).scatter_(2, traj_actions.long(), addition)
         
                    if th.max(traj_latents.detach()) > 50 or th.max(traj_latents.detach()) < -50:
                        print("latents abs mean before clamp: {}, max: {}, min: {}".format(th.mean(th.abs(traj_latents.detach())), th.max(traj_latents.detach()), th.min(traj_latents.detach())))

                    traj_latents = traj_latents.clamp(min=-50, max=50).detach()



                old_distribution = Categorical(probs=F.softmax(traj_old_latents.view(-1, self.action_space.n), dim=1))
                    
                new_distribution = Categorical(probs=F.softmax(traj_latents.view(-1, self.action_space.n).detach(), dim=1))

                reg_loss = self.reg_coef * th.norm(traj_old_latents.view(-1, self.action_space.n), dim=1, p=2).mean()
                ent_loss = self.ent_coef * old_distribution.entropy().mean()
                policy_loss = th.distributions.kl_divergence(old_distribution, new_distribution).mean() + reg_loss + ent_loss
                
                # print(reg_loss.requires_grad)
                # print(policy_loss.requires_grad)
                if i_gradient_step == 0 and True:
                    print("max traj_latents: ", th.max(traj_latents.flatten(),dim=-1))
                    print("Old max prob: ", th.max(old_distribution.probs))
                    print("New max prob: ", th.max(new_distribution.probs))
                    print("Max difference: ", th.max(th.abs(new_distribution.probs - old_distribution.probs)))
                    print("regularization loss: ", reg_loss) 
                    print("Ent loss: ", ent_loss) 
                    print("KL loss: ", policy_loss - reg_loss)


                policy_loss = 0.0
                for i in range(num):
                    
                    # print('t: ', traj_latents[i].size())
                    old_distribution = Categorical(probs=F.softmax(traj_old_latents[i], dim=1))
                    
                    new_distribution = Categorical(probs=F.softmax(traj_latents[i].detach(), dim=1))
                    policy_loss += th.distributions.kl_divergence(old_distribution, new_distribution).sum()
                
                policy_loss /= num
            
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
            # value_loss = 0.0
            # policy_loss = 0.0
            # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            for rollout_data in self.rollout_buffer.get(batch_size=None):

                actions = rollout_data.actions
                actions = actions.long().flatten()
                advantages = rollout_data.advantages

                
                # values = rollout_data.old_values
                if self.use_v_net:
                    latent_pi, latent_vf, latent_sde = self.policy._get_latent(rollout_data.observations)
                    values = self.value_net(latent_vf)
                else:
                    values = self.policy.compute_value(rollout_data.observations)

                values = values.flatten()
                log_probs = self.policy.get_action_log_probs(rollout_data.observations, actions.unsqueeze(1))
                assert advantages.size() == log_probs.flatten().size()
                on_policy_value_loss = F.mse_loss(rollout_data.returns, values)
                on_policy_policy_loss = -(advantages * log_probs.flatten()).mean()

            # on_policy_value_loss=0.0
            # print(on_policy_policy_loss.requires_grad)
            # print(value_loss)
            # print(on_policy_value_loss.requires_grad)
            # loss = policy_loss + on_policy_policy_loss + self.vf_coef * (value_loss + on_policy_value_loss) / 2
            
            loss = on_policy_policy_loss + self.vf_coef * (on_policy_value_loss)  # a2c
            # loss = policy_loss + self.vf_coef * (value_loss) 
            # loss = on_policy_policy_loss
            # print(th.sum(th.isinf(loss)))
            if th.sum(th.isinf(loss)) > 0:
                print("min alpha: ", th.min(alpha))
                print("max alpha: ", th.max(alpha))
                print("min latent: ", th.min(traj_latents))
                print("max latent: ", th.max(traj_latents))
                print("INF detected in loss")
                print("policy_loss: ", policy_loss)
                print("vf_coef * value_loss: ", self.vf_coef * value_loss)
                exit(-1)
            # loss=policy_loss
            
            
            # Optimization step
            # print("value_loss 2: ",value_loss)
            # print("policy_loss 2: ", policy_loss / num)
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.n_backward += 1
            
            # print(loss)
            # print(self.policy.action_net)
            # print(self.policy.action_net.weight.grad)
            
            # # print(self.policy.action_net.weight.grad)
            # print(loss)

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.policy.optimizer.step()
            
            '''
            /////
            '''
        self._n_updates += 1
        self._on_update()
        

        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/policy_loss", np.mean(policy_losses))
        logger.record("rollout/epsilon", self.exploration_rate)
        
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
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
