from typing import Any, Dict, Optional, Type, Union, Tuple
import torch as th
import gym
import os, sys 
sys.path.insert(0, os.environ['SB_PATH'])
from gym import spaces
from torch.nn import functional as F
from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.offcapo.policies import OffCAPOMlpPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import explained_variance, polyak_update, get_linear_fn, is_vectorized_observation
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import TrajectoryBuffer, RetraceBuffer
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps

from torch.distributions import Categorical


class OffCAPO(OffPolicyAlgorithm):

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        buffer_size: int = 1000000,
        # n_steps: int = 10,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 0.9,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        alpha_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        train_freq: Union[int, Tuple[int, str]] = (128, 'step'),
        gradient_steps: int = 1,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        create_eval_env: bool = False,
        _init_setup_model: bool = True,
        exploration_fraction: float = 0.5,
        exploration_initial_eps: float = 0.5,
        exploration_final_eps: float = 0.01,
        support_multi_env: bool = True,
        max_alpha: int = 10,
        reg_coef: float = 0.0,
        use_rms_prop: bool = False,
        rms_prop_eps: float = 1e-5,
        policy_delay: int = 2
    ):

        super(OffCAPO, self).__init__(
            policy,
            env,
            OffCAPOMlpPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=TrajectoryBuffer,
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

        )


        self.policy_delay = policy_delay
        self.reg_coef = reg_coef
        self.max_alpha = max_alpha
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.alpha_coef = alpha_coef
        self.max_grad_norm = max_grad_norm
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = exploration_initial_eps
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        # self.n_steps = n_steps
        self.retrace_buffer = None
        '''
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        
        '''
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)
        if _init_setup_model:
            self._setup_model()

    
        
            
    def _setup_model(self) -> None:
        super(OffCAPO, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )
        assert self.train_freq.unit == TrainFrequencyUnit.STEP, "OffCAPO only train on steps"

        self.retrace_buffer = RetraceBuffer(
            self.train_freq.frequency, 
            self.observation_space, 
            self.action_space,
            self.device,
            self.gamma,
            self.n_envs
        )


    def _create_aliases(self) -> None:
        self.action_net = self.policy.action_net
        self.q_net = self.policy.q_net
        self.q_net_target =self.policy.q_net_target
        # self.action_target = self.policy.action_target
        # self.critic = self.policy.critic
        # self.critic_target = self.policy.critic_target


    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        use_target: bool = True
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
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state


    
    def _on_step(self) -> None:
        """
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def _on_update(self) -> None:
        pass
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: TrajectoryBuffer,
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
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            # self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
                
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()
            next_obs = new_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    next_obs[idx] = terminal_obs

            self.retrace_buffer.add(self._last_obs, actions, rewards, next_obs, dones)
            self._last_obs = new_obs

        _obs, _actions, _rewards , _next_obs, _dones = self.retrace_buffer.get_trajectories()
        replay_buffer.add(
            _obs, _actions, _rewards, _next_obs, _dones
        )
        self.retrace_buffer.reset()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int=100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        # self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        # TODO separate learning rate
        self._update_learning_rate([self.policy.optimizer])

        actor_losses, critic_losses, rhos_l = [], [], []

        for _ in range(gradient_steps):
            
            losses = 0
            self._n_updates += 1
            # Sample replay buffer
            assert self._vec_normalize_env is None, "Not Implemented yet"

            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            actual_batch_size = replay_data.observations.shape[0]
            Q_ret = th.zeros(replay_data.rewards.shape)
            assert self.train_freq.frequency == replay_data.actions.shape[1]
            # TODO
            thetas = self.policy.get_theta(replay_data.observations).reshape(-1, self.action_space.n)
            probs = F.softmax(thetas, dim=1)
            actions = replay_data.actions.reshape((-1, ) +  self.action_space.shape)
            probs = th.gather(probs, dim=1, index=actions.unsqueeze(-1)).reshape((actual_batch_size ,self.train_freq.frequency, -1)).squeeze()
            # print(probs.shape)
            # print(Q_ret[:, 1].shape)
            # exit()
            ret_cs = th.ones((actual_batch_size), dtype=th.float64)
            r = 0

            # exit()
            for idx in reversed(range(self.train_freq.frequency)):
                with th.no_grad():
                    if idx == self.train_freq.frequency - 1:
                        
                        next_Q = self.policy.compute_value(replay_data.next_observations[:, -1], use_target=True).squeeze()
                        # print(next_Q)
                    else:
                        next_Q = Q_ret[:, idx+1]
                    Q_ret[:, idx] = replay_data.rewards[:, idx] + 1 * self.gamma * (1-replay_data.dones[:, idx]) * next_Q
                    r = replay_data.rewards[0, idx] + self.gamma * r
                    # assert ret_cs.shape == probs[:, idx].shape and  ret_cs.shape  == Q_ret[:, idx].shape
                    # ret_cs = (ret_cs * probs[:, idx]).clamp(max=1)

                    
            # print(Q_ret[:, idx])
            # print(r)
            # # Get current Q-values estimates for each critic network
            current_q_values = self.policy.compute_Q(replay_data.observations, replay_data.actions, use_target=False)
            # Compute Advantage
            with th.no_grad():

                current_v_values = self.policy.compute_value(replay_data.observations, use_target=True)
                assert current_v_values.shape == current_q_values.shape, f"current_v_values.shape: {current_v_values.shape}"

                advantages = current_q_values - current_v_values
                
                assert advantages.shape == current_q_values.shape, f"advantages.shape: {advantages.shape}"


            # # Compute critic loss
            assert current_q_values.shape == Q_ret.shape
            critic_loss = F.smooth_l1_loss(current_q_values.flatten(), Q_ret.flatten(), reduction='mean') * self.vf_coef
            # critic_loss = F.mse_loss(current_q_values.flatten(), Q_ret.flatten(), reduction='mean') * self.vf_coef
            
            # print("current: ", current_q_values.flatten()[0:10])
            # print("Q_ret: ", Q_ret.flatten()[0:10])
            critic_losses.append(critic_loss.item())

            # # Optimize the critics
            losses += critic_loss 
            policy_loss = None
            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                
                with th.no_grad():
                    thetas_hat = thetas.detach().clone()
                    probs = F.softmax(thetas_hat, dim=1)

                    actions = replay_data.actions.reshape((-1, ) +  self.action_space.shape)
                    probs = th.gather(probs, dim=1, index=actions.unsqueeze(-1))
                    # reshape actions from (batchsize) => (batchsize, 1) to scatter
                    alpha = -1 * th.log(probs + 1e-10) * self.alpha_coef 
                    advantages = advantages.reshape(-1, 1)
                    assert alpha.shape == advantages.shape, f"alpa.shape={alpha.shape}, advantage.shape={advantages.shape}"
                    addition = alpha * th.sign(advantages)
                    thetas_hat = thetas_hat + th.zeros_like(thetas_hat).scatter_(1, actions.reshape(-1,1).long(), addition.reshape(-1, 1)) 
                    thetas_hat.clamp(min=-100, max=100)

                old_distribution = Categorical(probs=F.softmax(thetas, dim=1))
                # print(old_distribution.probs)
                new_distribution = Categorical(probs=F.softmax(thetas_hat.detach(), dim=1))
                # policy_loss = th.distributions.kl_divergence(old_distribution, new_distribution).mean()
                policy_loss = th.distributions.kl_divergence(new_distribution, old_distribution).mean()
                losses += policy_loss
                self.logger.record("train/policy_loss", policy_loss.item())
                self.logger.record("train/alpha_mean", alpha.mean().item())
                self.logger.record("train/alpha_max", alpha.max().item())
                self.logger.record("train/alpha_min", alpha.min().item())
                self.logger.record("train/prob_max", probs.max().item())
                self.logger.record("train/prob_mean", probs.mean().item())
                self.logger.record("train/prob_min", probs.min().item())
                self.logger.record("train/thetas_hat_max", thetas_hat.max().item())
            # Optimize the actor
            # self.actor.optimizer.zero_grad()
            # actor_loss.backward()
            # self.actor.optimizer.step()

            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # polyak_update(self.action_net.parameters(), self.action_target.parameters(), self.tau)
            
            self.policy.optimizer.zero_grad()
            losses.backward()
            self.policy.optimizer.step()
            # self.critic.optimizer.zero_grad()
            # critic_loss.backward()
            # self.critic.optimizer.step()

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            # self.logger.record("train/rho_mean", np.mean(rhos_l))
            
            self.logger.record("train/value_loss", critic_loss.item())
        
            self._on_update()
        
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 50,
        tb_log_name: str = "OffPAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OffCAPO":

        
        rv = super(OffCAPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
        return rv
