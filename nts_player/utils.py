import jax.numpy as jnp
from jax import vmap
import json

import flax
import flax.linen as nn

from turing_vs_scherbius import PyGameState

def construct_observation_from_state(state: PyGameState) -> jnp.ndarray:
    """
    
    """
    config = state.game_config
    max_cards = config.max_cards
    max_card_value = config.max_card_value

class MLP(nn.Module):
    player_name: str
    max_cards: int
    max_card_value: int
    max_vp: int
    n_battles: int

    width: int

    def setup(self):
        self.card_embeddings = nn.Embed(num_embeddings=self.max_card_value, features=self.width)
        self.vp_embeddings = nn.Embed(num_embeddings=self.max_vp, features=self.width)

    def __call__(self, observation, rewards, cards):

        card_rewards, vp_rewards = rewards

        # 
        cr_embeddings = vmap(self.card_embeddings)(card_rewards)
        vpr_embeddings = self.vp_embeddings(vp_rewards)

