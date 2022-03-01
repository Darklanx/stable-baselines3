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
from stable_baselines3.offpac.policies import OffPACPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import explained_variance, polyak_update, get_linear_fn, is_vectorized_observation
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
import time
from geomloss import SamplesLoss


class OffPAC(OffPolicyAlgorithm):

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 0.9,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
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
        )

 
        self.policy_delay = policy_delay
        self.reg_coef = reg_coef
        self.max_alpha = max_alpha
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = exploration_initial_eps
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
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
        super(OffPAC, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )


    def _create_aliases(self) -> None:
        self.action_net = self.policy.action_net
        self.action_target = self.policy.action_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target


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

    def _on_update(self) -> None:
        pass


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
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                # noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                # noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                # next_actions = (self.actor_target(replay_data.next_observations)).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                # next_q_values = th.cat(self.critic_target(replay_data.next_observations, replay_data.actions), dim=1)
                next_v_values = self.policy.compute_value(replay_data.next_observations, use_target=True).view(-1, 1)
                
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v_values

                
                

            # # Get current Q-values estimates for each critic network
            current_q_values = self.policy.get_Q(replay_data.observations, replay_data.actions)

            # Compute Advantage
            with th.no_grad():
                current_v_values = self.policy.compute_value(replay_data.observations, use_target=True).view(-1, 1)
                assert current_v_values.shape == (self.batch_size, 1), f"current_v_values.shape: {current_v_values.shape}"

                advantages = current_q_values - current_v_values
                assert advantages.shape == (self.batch_size, 1), f"advantages.shape: {advantages.shape}"
                    


            # # Compute critic loss
            critic_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='mean') * self.vf_coef
            critic_losses.append(critic_loss.item())

            # # Optimize the critics
            losses += critic_loss 

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                # actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                

                log_probs = self.policy.get_action_log_probs(replay_data.observations, replay_data.actions, use_target=False)
                # print(log_probs.requires_grad)
                # exit()
                assert log_probs.shape == (self.batch_size, 1), f"log_probs.shape: {log_probs.shape}"
                behav_log_probs = self.policy.get_action_log_probs(replay_data.observations, replay_data.actions, use_target=True)
                rhos = (log_probs.detach() - behav_log_probs.detach()).exp()
                rhos_l.append(rhos.mean())
                actor_loss = (-1 * rhos.detach() * advantages.detach() * log_probs).mean()
                losses += actor_loss
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                # self.actor.optimizer.zero_grad()
                # actor_loss.backward()
                # self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.action_net.parameters(), self.action_target.parameters(), self.tau)
            
            self.policy.optimizer.zero_grad()
            losses.backward()
            self.policy.optimizer.step()
            # self.critic.optimizer.zero_grad()
            # critic_loss.backward()
            # self.critic.optimizer.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/rho_mean", np.mean(rhos_l))
        if len(actor_losses) > 0:
            self.logger.record("train/policy_loss", np.mean(actor_losses))
        self.logger.record("train/value_loss", np.mean(critic_losses))
        
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
    ) -> "OffPAC":

        
        rv = super(OffPAC, self).learn(
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
