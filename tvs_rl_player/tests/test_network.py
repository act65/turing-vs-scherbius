# test_network.py
import pytest
import jax
import jax.numpy as jnp
import numpy as np # For creating numpy arrays for observations easily
import flax.linen as nn

# Assuming these are in a module, e.g., 'tvs_rl.network'
# from tvs_rl.network import GameAndNetworkConfig, TvSNetAutoregressive
# For this example, I'll redefine them briefly or assume they are in scope.

# --- Minimal Redefinitions for testing context ---
from dataclasses import dataclass

@dataclass
class GameAndNetworkConfig:
    n_battles: int
    max_cards_per_battle_strategy: int
    card_vocab_size: int # Max card value (e.g., 10)
    max_hand_size: int

    card_embed_dim: int = 16 # Smaller for tests
    lstm_hidden_size: int = 32 # Smaller for tests
    mlp_hidden_size: int = 24 # Smaller for tests

    @property
    def card_vocab_size_for_embedding(self) -> int:
        return self.card_vocab_size + 1 # For padding/0-value

    @property
    def num_card_types_for_q_values(self) -> int: # Not used in autoregressive head
        return self.card_vocab_size + 1

# Using the TvSNetAutoregressive structure from the previous response
# (Make sure it's correctly defined and accessible)
# For brevity, I won't redefine the full TvSNetAutoregressive here.
# Assume it's imported or defined above this test file.
# --- End Minimal Redefinitions ---


# Dummy TvSNetAutoregressive for testing if not importing
class TvSNetAutoregressive(nn.Module):
    config: GameAndNetworkConfig
    player_perspective: str

    def setup(self):
        self.card_embed = nn.Embed(
            num_embeddings=self.config.card_vocab_size_for_embedding,
            features=self.config.card_embed_dim
        )
        self.hand_mlp = nn.Dense(features=self.config.mlp_hidden_size)
        self.card_rewards_mlp = nn.Dense(features=self.config.mlp_hidden_size)
        if self.player_perspective == "Turing":
            self.intercepted_mlp = nn.Dense(features=self.config.mlp_hidden_size)
        self.scalar_mlp = nn.Dense(features=self.config.mlp_hidden_size)
        self.lstm_cell = nn.LSTMCell(features=self.config.lstm_hidden_size)
        
        # Simplified action_slot_decoder_mlp for testing structure
        self.action_slot_decoder_mlp_layers = [
            nn.Dense(features=self.config.mlp_hidden_size), nn.relu,
            nn.Dense(features=self.config.max_hand_size + 1)
        ]
        if self.player_perspective == "Scherbius":
            self.reencrypt_decoder_mlp_layers = [
                nn.Dense(features=self.config.mlp_hidden_size // 2), nn.relu,
                nn.Dense(features=2)
            ]

    def encode(self, observations: dict, hidden_state):
        batch_size = observations['my_hand'].shape[0]
        my_hand_cards = observations['my_hand']
        hand_emb = self.card_embed(my_hand_cards)
        processed_hand = nn.relu(self.hand_mlp(jnp.sum(hand_emb, axis=1)))

        last_card_rewards = observations['last_round_card_rewards']
        card_rewards_emb = self.card_embed(last_card_rewards)
        flat_card_rewards = card_rewards_emb.reshape(batch_size, -1)
        processed_card_rewards = nn.relu(self.card_rewards_mlp(flat_card_rewards))
        
        processed_intercepted_list = []
        if self.player_perspective == "Turing":
            intercepted_strat = observations['intercepted_scherbius_strategy']
            intercepted_emb = self.card_embed(intercepted_strat)
            flat_intercepted = intercepted_emb.reshape(batch_size, -1)
            processed_intercepted = nn.relu(self.intercepted_mlp(flat_intercepted))
            processed_intercepted_list.append(processed_intercepted)

        scalar_features = [
            jnp.expand_dims(observations['my_points'], axis=-1),
            jnp.expand_dims(observations['opponent_points'], axis=-1),
            observations['last_round_vp_rewards']
        ]
        concatenated_scalars = jnp.concatenate(scalar_features, axis=-1)
        processed_scalars = nn.relu(self.scalar_mlp(concatenated_scalars))
        
        lstm_input_list = [processed_hand, processed_card_rewards, processed_scalars] + processed_intercepted_list
        lstm_input = jnp.concatenate(lstm_input_list, axis=-1)
        new_hidden_state, lstm_output = self.lstm_cell(hidden_state, lstm_input)
        return lstm_output, new_hidden_state

    def decode_action_slot(self, lstm_output): # Simplified for test
        x = lstm_output
        for layer in self.action_slot_decoder_mlp_layers:
            x = layer(x)
        return x

    def decode_reencrypt(self, lstm_output): # Simplified for test
        x = lstm_output
        for layer in self.reencrypt_decoder_mlp_layers:
            x = layer(x)
        return x

    def initial_state(self, batch_size: int):
        return self.lstm_cell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), self.config.lstm_hidden_size)


@pytest.fixture
def turing_config():
    return GameAndNetworkConfig(
        n_battles=2, max_cards_per_battle_strategy=2,
        card_vocab_size=10, max_hand_size=5,
        card_embed_dim=8, lstm_hidden_size=16, mlp_hidden_size=12
    )

@pytest.fixture
def scherbius_config():
    return GameAndNetworkConfig(
        n_battles=2, max_cards_per_battle_strategy=2,
        card_vocab_size=10, max_hand_size=5,
        card_embed_dim=8, lstm_hidden_size=16, mlp_hidden_size=12
    )

def get_dummy_obs(batch_size, config: GameAndNetworkConfig, for_turing: bool):
    obs = {
        'my_hand': jnp.zeros((batch_size, config.max_hand_size), dtype=jnp.int32),
        'my_points': jnp.zeros(batch_size, dtype=jnp.int32),
        'opponent_points': jnp.zeros(batch_size, dtype=jnp.int32),
        'last_round_card_rewards': jnp.zeros((batch_size, config.n_battles, config.max_cards_per_battle_strategy), dtype=jnp.int32),
        'last_round_vp_rewards': jnp.zeros((batch_size, config.n_battles), dtype=jnp.int32),
    }
    if for_turing:
        obs['intercepted_scherbius_strategy'] = jnp.zeros((batch_size, config.n_battles, config.max_cards_per_battle_strategy), dtype=jnp.int32)
    return obs

def test_network_initialization_turing(turing_config):
    key = jax.random.PRNGKey(0)
    model = TvSNetAutoregressive(config=turing_config, player_perspective="Turing")
    batch_size = 1
    dummy_obs = get_dummy_obs(batch_size, turing_config, for_turing=True)
    initial_lstm_state = model.initial_state(batch_size)
    
    params = model.init(key, dummy_obs, initial_lstm_state, method=model.encode)['params']
    assert params is not None
    # Check a few param names exist
    assert 'card_embed' in params
    assert 'lstm_cell' in params
    assert 'intercepted_mlp' in params # Turing specific
    assert 'action_slot_decoder_mlp_layers_1' in params # decoder dense layer

def test_network_initialization_scherbius(scherbius_config):
    key = jax.random.PRNGKey(0)
    model = TvSNetAutoregressive(config=scherbius_config, player_perspective="Scherbius")
    batch_size = 1
    dummy_obs = get_dummy_obs(batch_size, scherbius_config, for_turing=False)
    initial_lstm_state = model.initial_state(batch_size)
    
    params = model.init(key, dummy_obs, initial_lstm_state, method=model.encode)['params']
    assert params is not None
    assert 'reencrypt_decoder_mlp_layers_1' in params # Scherbius specific
    assert 'intercepted_mlp' not in params.get('params', params) # Should not exist for Scherbius

def test_encode_method(turing_config):
    key = jax.random.PRNGKey(42)
    model = TvSNetAutoregressive(config=turing_config, player_perspective="Turing")
    batch_size = 2
    dummy_obs = get_dummy_obs(batch_size, turing_config, for_turing=True)
    initial_lstm_state = model.initial_state(batch_size)
    params = model.init(key, dummy_obs, initial_lstm_state, method=model.encode)['params']

    context_vector, new_lstm_state = model.apply(
        {'params': params}, dummy_obs, initial_lstm_state, method=model.encode
    )
    assert context_vector.shape == (batch_size, turing_config.lstm_hidden_size)
    assert new_lstm_state[0].shape == initial_lstm_state[0].shape # c_state
    assert new_lstm_state[1].shape == initial_lstm_state[1].shape # h_state
    assert not jnp.array_equal(new_lstm_state[1], initial_lstm_state[1]) # h_state should change

def test_decode_action_slot_method(turing_config):
    key = jax.random.PRNGKey(43)
    model = TvSNetAutoregressive(config=turing_config, player_perspective="Turing")
    batch_size = 2
    
    # Dummy context vector (output of encode)
    dummy_context = jnp.zeros((batch_size, turing_config.lstm_hidden_size))
    
    # Initialize only the decoder part for this test if possible, or full model
    # For simplicity, init full model then call decoder.
    dummy_obs = get_dummy_obs(batch_size, turing_config, for_turing=True) # For init
    initial_lstm_state = model.initial_state(batch_size) # For init
    params = model.init(key, dummy_obs, initial_lstm_state, method=model.encode)['params'] # Get all params

    q_values = model.apply(
        {'params': params}, dummy_context, method=model.decode_action_slot
    )
    
    expected_q_shape = (batch_size, turing_config.max_hand_size + 1)
    assert q_values.shape == expected_q_shape

def test_decode_reencrypt_method(scherbius_config):
    key = jax.random.PRNGKey(44)
    model = TvSNetAutoregressive(config=scherbius_config, player_perspective="Scherbius")
    batch_size = 2
    dummy_context = jnp.zeros((batch_size, scherbius_config.lstm_hidden_size))

    dummy_obs = get_dummy_obs(batch_size, scherbius_config, for_turing=False)
    initial_lstm_state = model.initial_state(batch_size)
    params = model.init(key, dummy_obs, initial_lstm_state, method=model.encode)['params']

    q_values_reencrypt = model.apply(
        {'params': params}, dummy_context, method=model.decode_reencrypt
    )
    expected_q_shape = (batch_size, 2) # Q(False), Q(True)
    assert q_values_reencrypt.shape == expected_q_shape