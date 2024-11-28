import jax.numpy as jnp

import flax
import flax.linen as nn

class NTSPlayer():
    def __init__(self, game_config, name):
        self.game_config = game_config
        self.name = name
        self.params = None
        self.optimizer = None
        self.step_counter = 0

    def load(self, path):
        pass

    def save(self, path):
        pass

    def select_action(self, observation, rewards, intercepted_scherbius_hand):
        pass

    def update(self, experience):
        pass



class NTSNetwork(nn.Module):

    name: str

    def apply(self, past_hands, past_rewards, past_intercepted_hand=None):
        """
        past_hands: jnp.ndarray (T, M, N)
        past_rewards: (jnp.ndarray (T, K, N), jnp.ndarray (T, 1))
        past_intercepted_hand: jnp.ndarray (T, L, N) or None
        """

        if self.name == 'turing':
            assert intercepted_opponent_hand is not None

