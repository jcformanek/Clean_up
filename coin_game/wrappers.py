""" Wrappers for use with jaxmarl baselines. """
import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial

# from gymnax.environments import environment, spaces
from typing import Union, Any

class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    # def _batchify(self, x: dict):
    #     x = jnp.stack([x[a] for a in self._env.agents])
    #     return x.reshape((self._env.num_agents, -1))

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    episode_coins: int
    returned_episode_coins: int
    episode_team_coins: int
    returned_episode_team_coins: int


class LogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths.

    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,), dtype=jnp.int32),
            jnp.zeros((self._env.num_agents,), dtype=jnp.int32),
            jnp.zeros((self._env.num_agents,), dtype=jnp.int32),
            jnp.zeros((self._env.num_agents,), dtype=jnp.int32),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        
        # Extract cumulative coin information from environment info
        # This tracks total coins collected by each agent (always increasing)
        cumulative_own = info.get("cumulative_own_coins_collected", jnp.zeros((self._env.num_agents,), dtype=jnp.int32))
        cumulative_other = info.get("cumulative_other_coins_collected", jnp.zeros((self._env.num_agents,), dtype=jnp.int32))
        
        # Calculate total cumulative coins per agent (own + other coins collected)
        current_coins = cumulative_own + cumulative_other
        
        # Calculate team total coins (sum across all agents)
        current_team_coins = jnp.sum(current_coins)
        # Broadcast team total to all agents for consistent logging
        current_team_coins_per_agent = jnp.full((self._env.num_agents,), current_team_coins, dtype=jnp.int32)

        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
            episode_coins=current_coins * (1 - ep_done),
            returned_episode_coins=state.returned_episode_coins * (1 - ep_done)
            + current_coins * ep_done,
            episode_team_coins=current_team_coins_per_agent * (1 - ep_done),
            returned_episode_team_coins=state.returned_episode_team_coins * (1 - ep_done)
            + current_team_coins_per_agent * ep_done,
        )

        if self.replace_info:
            info = {}

        info["episode_returns"] = state.returned_episode_returns
        info["episode_lengths"] = state.returned_episode_lengths
        info["episode_coins"] = state.returned_episode_coins
        info["episode_team_coins"] = state.returned_episode_team_coins
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        
        return obs, state, reward, done, info