from typing import Any, Dict, Optional, Type, Union, Tuple

import torch as th
import gym
from gym import spaces
from torch.nn import functional as F
from torch.distributions import Bernoulli, Categorical, Normal
from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.offpac.policies import OFFPACPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
import numpy as np

class OFFPAC(OffPolicyAlgorithm):

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 0.005,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
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
        KL: bool = False
    ):

        super(OFFPAC, self).__init__(
            policy,
            env,
            OFFPACPolicy,
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
        )
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.KL = KL
        '''
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)
        '''
        if _init_setup_model:
            self._setup_model()


    def train(self, gradient_steps: int, batch_size: int=100) -> None:
        self._update_learning_rate(self.policy.optimizer)
        value_losses = []
        policy_losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                # V(s+1)
                next_values = self.policy.compute_value(replay_data.next_observations).view(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_values
            
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values, log_prob, entropy  = self.policy.evaluate_actions(replay_data.observations, replay_data.actions)
            
            # A(s,a) = Q(s,a) - V(s)
            current_values = self.policy.compute_value(replay_data.observations).view(-1, 1)
            advantages = current_q_values - current_values

            value_loss = F.mse_loss(current_q_values, target_q_values)
            value_losses.append(value_loss.item())
            # print("target_q_values: ", target_q_values.size())
            # print("adv: ", advantages.size())
            # print("log_prob: ", log_prob.size())
            if not self.KL:
                policy_loss = -(advantages.detach() * log_prob).mean()
            else:
                latent, old_distribution = self.policy.get_policy_latent(replay_data.observations)
                latent = latent.detach()
                with th.no_grad():
                    action_prob = th.exp(log_prob)
                    alpha  = 1.0 / action_prob / 10
                    alpha = th.clamp(alpha, -10, 10)
                    _j = th.arange(latent.size(0)).long()
                    action_idx = replay_data.actions.squeeze()
                    latent[_j, action_idx] += (th.sign(advantages.detach()) * alpha * (1-action_prob)).squeeze()
                    latent = latent - th.mean(latent, dim=1)[0].view(-1,1)
                    latent = th.clamp(latent, -50, 50)

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
            policy_losses.append(policy_loss.item())


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

        self._n_updates += 1

        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/policy_loss", np.mean(policy_losses))
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
        log_interval: int = 100,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OFFPAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OFFPAC":

        return super(OFFPAC, self).learn(
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
