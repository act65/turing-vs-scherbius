# agent.py
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn # For type hinting if TvSNetAutoregressive is nn.Module

# Assuming TvSNetAutoregressive and GameAndNetworkConfig are importable
# from tvs_rl.network import TvSNetAutoregressive, GameAndNetworkConfig

class Agent:
    def __init__(self,
                 model: TvSNetAutoregressive, # The Flax model instance
                 params: dict,                 # Model parameters
                 player_perspective: str,
                 net_config: GameAndNetworkConfig,
                 seed: int = 0):
        self.model = model
        self.params = params
        self.player_perspective = player_perspective
        self.net_config = net_config
        self.key_base = jax.random.PRNGKey(seed)
        self.key_counter = 0

        # LSTM state (c_state, h_state)
        self.lstm_state = None
        self.reset() # Initialize LSTM state

    def _get_key(self):
        self.key_counter += 1
        return jax.random.fold_in(self.key_base, self.key_counter)

    def reset(self):
        # Batch size is 1 for action selection by a single agent
        self.lstm_state = self.model.initial_state(batch_size=1)
        self.key_counter = 0 # Reset key sequence for reproducibility if desired

    def _preprocess_observation(self, observation: dict) -> dict:
        """Adds batch dimension and ensures JAX arrays for network input."""
        jax_obs = {}
        for key, value in observation.items():
            if isinstance(value, (np.ndarray, list, int, float)): # Common types from env
                val_arr = np.array(value)
                # Ensure correct dtype, especially for indices (int32)
                if val_arr.dtype == np.float64: val_arr = val_arr.astype(np.float32)
                if val_arr.dtype == np.int64: val_arr = val_arr.astype(np.int32)
                
                jax_obs[key] = jnp.expand_dims(val_arr, axis=0) # Add batch dim
            else:
                jax_obs[key] = value # Pass through if already exotic
        return jax_obs

    def _postprocess_action(self,
                            chosen_strategy_hand_indices: np.ndarray, # (n_battles, max_cards_per_battle)
                            chosen_reencrypt: bool, # For Scherbius
                            current_hand_cards: list[int] # Actual card values in hand
                           ) -> tuple:
        """Converts chosen hand indices to actual card values for the environment."""
        
        # Create the strategy matrix with actual card values
        # chosen_strategy_hand_indices contains original hand indices, or -1 for "no card"
        
        env_strategy = [[] for _ in range(self.net_config.n_battles)]
        
        # Map hand indices to card values
        # This assumes chosen_strategy_hand_indices stores the *index* from the original hand
        # or a sentinel for "no card" (e.g., self.net_config.max_hand_size)
        
        for battle_idx in range(self.net_config.n_battles):
            for slot_idx in range(self.net_config.max_cards_per_battle_strategy):
                hand_card_idx = chosen_strategy_hand_indices[battle_idx, slot_idx]
                if 0 <= hand_card_idx < len(current_hand_cards):
                    env_strategy[battle_idx].append(current_hand_cards[hand_card_idx])
                # If hand_card_idx is the "no card" sentinel or out of bounds, nothing is added.
                # The environment expects empty lists or lists of cards for battles.

        if self.player_perspective == "Scherbius":
            return env_strategy, chosen_reencrypt
        else:
            return env_strategy, None # Turing doesn't re-encrypt

    @jax.jit # Jit the core computation for speed
    def _apply_model_encode(self, params, obs, lstm_state):
        return self.model.apply({'params': params}, obs, lstm_state, method=self.model.encode)

    @jax.jit
    def _apply_model_decode_slot(self, params, context_vector):
        return self.model.apply({'params': params}, context_vector, method=self.model.decode_action_slot)
    
    @jax.jit
    def _apply_model_decode_reencrypt(self, params, context_vector): # Scherbius only
        return self.model.apply({'params': params}, context_vector, method=self.model.decode_reencrypt)

    def select_action(self,
                      observation: dict,       # Raw observation from environment
                      current_hand_cards: list[int], # Actual cards in hand, e.g., [1, 5, 10, 2, 8]
                      is_training: bool = True,
                      epsilon: float = 0.05):
        """
        Selects an action autoregressively.
        Returns:
            env_action: Action formatted for the TvSEnvironment.
            aux_data: Dictionary with 'q_values_for_chosen_slots', 'chosen_hand_indices' for training.
        """
        # 1. Preprocess observation and update LSTM state
        processed_obs = self._preprocess_observation(observation)
        
        context_vector, new_lstm_state = self._apply_model_encode(
            self.params, processed_obs, self.lstm_state
        )
        self.lstm_state = new_lstm_state # Update agent's recurrent state

        # --- Autoregressive strategy selection ---
        # Strategy matrix to be filled with *original hand indices*
        # (0 to max_hand_size-1 for a card, or max_hand_size for "no card")
        num_slots_total = self.net_config.n_battles * self.net_config.max_cards_per_battle_strategy
        
        # Stores the original hand index chosen for each slot, or self.net_config.max_hand_size for "no card"
        chosen_hand_indices_flat = np.full(num_slots_total, self.net_config.max_hand_size, dtype=int)
        
        # Mask of available cards from the *original* hand positions for the current turn's strategy
        # True if card at that original hand index is available.
        available_in_original_hand_mask = np.array([True] * len(current_hand_cards) + [False] * (self.net_config.max_hand_size - len(current_hand_cards)))
        
        # For storing Q-values of chosen actions for training
        q_values_for_chosen_slots = np.zeros(num_slots_total, dtype=np.float32)

        for slot_idx in range(num_slots_total):
            # Get Q-values for all possible original hand cards + "no card" option
            # q_for_slot has shape (1, max_hand_size + 1)
            q_for_slot_all_options = self._apply_model_decode_slot(self.params, context_vector)
            q_for_slot_all_options_np = np.array(q_for_slot_all_options[0]) # Remove batch dim

            # Create a mask for valid actions for *this specific slot*:
            # - Can choose any card from original hand *if it's still available*.
            # - Can always choose "no card" (the last Q-value).
            current_valid_actions_mask = np.append(available_in_original_hand_mask, True) # (max_hand_size + 1)
            
            masked_q_values = np.where(current_valid_actions_mask, q_for_slot_all_options_np, -np.inf)

            # Epsilon-greedy selection
            key = self._get_key()
            if is_training and np.random.rand() < epsilon: # Using numpy random for this part
                # Choose randomly from valid actions
                valid_action_indices = np.where(current_valid_actions_mask)[0]
                chosen_action_idx_for_slot = np.random.choice(valid_action_indices)
            else:
                chosen_action_idx_for_slot = np.argmax(masked_q_values)

            chosen_hand_indices_flat[slot_idx] = chosen_action_idx_for_slot
            q_values_for_chosen_slots[slot_idx] = q_for_slot_all_options_np[chosen_action_idx_for_slot]

            # If a real card was chosen (not "no card"), mark it as unavailable for subsequent slots
            if chosen_action_idx_for_slot < self.net_config.max_hand_size:
                if chosen_action_idx_for_slot < len(current_hand_cards): # Ensure it's a valid hand index
                     available_in_original_hand_mask[chosen_action_idx_for_slot] = False
                else: # Should not happen if logic is correct, means agent chose a padded hand slot
                    chosen_hand_indices_flat[slot_idx] = self.net_config.max_hand_size # Force "no card"

        # Reshape chosen_hand_indices to (n_battles, max_cards_per_battle_strategy)
        final_chosen_strategy_hand_indices = chosen_hand_indices_flat.reshape(
            (self.net_config.n_battles, self.net_config.max_cards_per_battle_strategy)
        )

        # --- Re-encrypt decision (Scherbius only) ---
        chosen_reencrypt = False
        q_value_for_reencrypt = None
        if self.player_perspective == "Scherbius":
            q_reencrypt = self._apply_model_decode_reencrypt(self.params, context_vector) # (1, 2)
            q_reencrypt_np = np.array(q_reencrypt[0])

            if is_training and np.random.rand() < epsilon: # Epsilon for re-encrypt
                chosen_reencrypt_idx = np.random.choice([0, 1])
            else:
                chosen_reencrypt_idx = np.argmax(q_reencrypt_np)
            
            chosen_reencrypt = bool(chosen_reencrypt_idx)
            q_value_for_reencrypt = q_reencrypt_np[chosen_reencrypt_idx]

        # 3. Postprocess action to environment format
        env_action_strategy, env_action_reencrypt = self._postprocess_action(
            final_chosen_strategy_hand_indices, chosen_reencrypt, current_hand_cards
        )

        aux_data = {
            "chosen_strategy_hand_indices": final_chosen_strategy_hand_indices, # For DQN target calculation
            "q_values_for_chosen_slots": q_values_for_chosen_slots, # For DQN loss
        }
        if self.player_perspective == "Scherbius":
            aux_data["chosen_reencrypt_action"] = chosen_reencrypt # For DQN target
            aux_data["q_value_for_reencrypt"] = q_value_for_reencrypt # For DQN loss
            env_action = (env_action_strategy, env_action_reencrypt)
        else:
            env_action = env_action_strategy


        return env_action, aux_data