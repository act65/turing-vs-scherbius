class TvSNetAutoregressive(nn.Module):
    config: GameAndNetworkConfig
    player_perspective: str

    def setup(self):
        # --- Encoder Part ---
        self.card_embed = nn.Embed(...)
        self.hand_mlp = nn.Dense(...)
        # ... other MLPs for preprocessing observation parts ...
        self.scalar_mlp = nn.Dense(...)
        self.lstm_cell = nn.LSTMCell(features=self.config.lstm_hidden_size, name="lstm_encoder_cell")

        # --- Decoder Part (Action Head for strategy slots) ---
        # This MLP takes (lstm_output, flattened_partial_strategy, current_slot_info)
        # and outputs Q-values for each card in the original hand + "no card".
        self.action_slot_decoder_mlp = [
            nn.Dense(features=self.config.mlp_hidden_size), nn.relu,
            nn.Dense(features=self.config.max_hand_size + 1) # Q for each hand pos + "no card"
        ]

        if self.player_perspective == "Scherbius":
            # This MLP takes (lstm_output, flattened_final_strategy)
            # and outputs Q-values for re-encryption.
            self.reencrypt_decoder_mlp = [
                nn.Dense(features=self.config.mlp_hidden_size // 2), nn.relu, # Smaller MLP
                nn.Dense(features=2) # Q(reencrypt=F), Q(reencrypt=T)
            ]

    def encode(self, observations: dict, hidden_state):
        # ... (similar to before: embed, process scalars, concat) ...
        # lstm_input = ...
        new_hidden_state, lstm_output = self.lstm_cell(hidden_state, lstm_input)
        return lstm_output, new_hidden_state

    def decode_action_slot(self, lstm_output, partially_built_strategy, available_cards_mask, current_slot_index):
        """
        Args:
            lstm_output: (batch, lstm_hidden_size) Context from encoder.
            partially_built_strategy: (batch, n_battles * max_cards_per_battle_strategy)
                                      Represents cards (by their original hand index or a special value)
                                      already placed. Padded with a special value for unassigned.
            available_cards_mask: (batch, max_hand_size) Boolean mask of available cards.
            current_slot_index: (batch,) Integer indicating which slot (0 to N_total_slots-1) is being decided.
        Returns:
            q_values_for_slot: (batch, max_hand_size + 1) Q-values for playing each original hand card
                               or "no card". Invalid actions (unavailable cards) should be masked outside.
        """
        # Flatten and combine inputs for the MLP
        # Maybe embed current_slot_index or use one-hot encoding
        # flattened_partial_strategy = partially_built_strategy.reshape(lstm_output.shape[0], -1)
        # slot_idx_one_hot = jax.nn.one_hot(current_slot_index, num_classes=self.config.n_battles * self.config.max_cards_per_battle_strategy)

        # decoder_input = jnp.concatenate([lstm_output, flattened_partial_strategy, slot_idx_one_hot], axis=-1)
        
        # Simplified: just use lstm_output for now, can add more context later
        # The complexity of what to feed the decoder for each slot can be iterated on.
        # For now, assume the lstm_output is rich enough.
        # A more sophisticated approach would involve attention over the hand cards,
        # conditioned on the lstm_output and the slot being filled.

        # Let's make a simpler first pass for the decoder:
        # It just uses lstm_output. The agent logic outside will handle masking.
        # This means the Q-values are for "intending" to play card from original_hand_slot_X.
        # The agent then checks if that card is available.
        # This is a middle ground. A better way is to feed the available_cards_mask
        # or embeddings of available cards directly into the decoder.

        # For a more robust Q-learning: Q(state, slot_idx, available_card_k)
        # This implies the decoder needs to know about available cards.
        # Let's assume for now the head outputs Q-values for *original hand positions*.
        # The agent applies the available_cards_mask *after* getting these Qs.

        x = lstm_output
        for layer in self.action_slot_decoder_mlp:
            x = layer(x)
        q_values_for_slot = x # (batch, max_hand_size + 1)
        return q_values_for_slot

    def decode_reencrypt(self, lstm_output, final_strategy): # Scherbius only
        # flattened_final_strategy = final_strategy.reshape(lstm_output.shape[0], -1)
        # decoder_input = jnp.concatenate([lstm_output, flattened_final_strategy], axis=-1)
        x = lstm_output # Simplified for now
        for layer in self.reencrypt_decoder_mlp:
            x = layer(x)
        q_values_reencrypt = x # (batch, 2)
        return q_values_reencrypt

    def initial_state(self, batch_size: int):
        return self.lstm_cell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), self.config.lstm_hidden_size)