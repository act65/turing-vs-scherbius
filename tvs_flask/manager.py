import tvs_core as tvs
import utils

class GameManager:
    def __init__(self, game_config):
        self.config = game_config
        self.state = utils.get_initial_state()
        self._scherbius_ai_fn = utils.scherbius_ai_player_logic
        self._turing_ai_fn = utils.turing_ai_player_logic

    def new_game(self, player_role):
        if player_role not in ["Turing", "Scherbius"]:
            return None, "Invalid player role specified."

        self.state = utils.get_initial_state()
        # Create the initial GameState object
        initial_gs_obj = tvs.GameState(self.config)
        # self.state["game_state"] will be properly set by prepare_round_start_data
        self.state["player_role"] = player_role
        
        # Pass the initial GameState object. is_new_round_for_player is True.
        client_data = self.prepare_round_start_data(initial_gs_obj, is_new_round_for_player=True)
        return client_data, None

    def get_current_game_state_for_client(self):
        current_gs_from_state = self.state.get("game_state")
        if not current_gs_from_state:
            return None, "Game not started. Please select a role to start a new game_state"

        # If data was already prepared for the current state and no action has occurred,
        # it can be returned. This is for refreshing the current view.
        if self.state.get("last_client_data_prepared"):
            # This assumes that if last_client_data_prepared is set, self.state['game_state']
            # and other relevant states are consistent with it.
            return self.state["last_client_data_prepared"], None
        
        if self.state["player_role"]:
             # Re-preparing data for a refresh. Use current game_state from self.state.
             # is_new_round_for_player=False means use existing player_initial_hand_for_turn for display.
             client_data = self.prepare_round_start_data(current_gs_from_state, is_new_round_for_player=False)
             return client_data, None
        return None, "Game state is inconsistent. Please start a new game_state"

    # Changed signature: added base_game_state parameter
    def prepare_round_start_data(self, base_game_state, is_new_round_for_player=True):
        game_state_to_process = base_game_state 
        player_role = self.state["player_role"]

        current_player_hand_values = []
        opponent_observed_plays = [[] for _ in range(self.config.n_battles)]
        scherbius_encryption_view = False
        current_phase = "Game_Over" # Default if game is over

        # This will be the game state after observations, or same as base_game_state if game over / no observations
        final_game_state_for_round_start = game_state_to_process 

        if not game_state_to_process.is_won:
            gs_after_obs, round_start_info = utils.handle_pre_turn_observations_and_ai(
                game_state_to_process, player_role, self.config, self._scherbius_ai_fn
            )
            final_game_state_for_round_start = gs_after_obs 
            current_phase = round_start_info["current_phase"]
            current_player_hand_values = round_start_info["current_player_hand_values"]
            opponent_observed_plays = round_start_info["opponent_observed_plays_for_player"]
            scherbius_encryption_view = round_start_info["scherbius_encryption_status_for_view"]

            if player_role == "Turing":
                self.state["scherbius_planned_strategy"] = round_start_info["scherbius_planned_strategy"]
                self.state["scherbius_planned_encryption"] = round_start_info["scherbius_planned_encryption"]
        else: # Game is won, populate hand values from the final state for display
            if player_role == "Turing":
                current_player_hand_values = final_game_state_for_round_start.turing_hand
            elif player_role == "Scherbius":
                current_player_hand_values = final_game_state_for_round_start.scherbius_hand
            # current_phase remains "Game_Over"
        
        player_hand_client, new_initial_hand = utils.prepare_player_hand_for_display(
            current_player_hand_values, # These are now correctly sourced
            self.state.get("player_initial_hand_for_turn"),
            is_new_round_for_player # This flag is crucial for how the hand is displayed/IDs are handled
        )
        self.state["player_initial_hand_for_turn"] = new_initial_hand
        
        card_rewards, vp_rewards = final_game_state_for_round_start.rewards
        self.state["current_round_potential_rewards"] = {"card_rewards": card_rewards, "vp_rewards": vp_rewards}

        client_data = utils.assemble_client_data_for_round_start(
            final_game_state_for_round_start, 
            player_role, self.config, self.state, # self.state is passed for history, last_summary etc.
            player_hand_client, opponent_observed_plays,
            scherbius_encryption_view, current_phase
        )

        # Removed: self.state["last_round_summary"] = None
        # last_round_summary is set in submit_player_action and should persist in client_data
        # until the next round is processed.

        self.state["last_client_data_prepared"] = client_data
        # Store the game state that represents the actual start of the player's turn (after observations)
        self.state['game_state'] = final_game_state_for_round_start
        return client_data

    def submit_player_action(self, player_submitted_strategy, scherbius_encrypts_from_player=None):
        game_state_before_action = self.state["game_state"]
        if not game_state_before_action or game_state_before_action.is_won:
            return None, "Game is over or not initialized."

        player_role = self.state["player_role"]

        if not isinstance(player_submitted_strategy, list) or \
           len(player_submitted_strategy) != self.config.n_battles:
            return None, "Invalid player strategy format."

        strategy_info = utils.determine_final_strategies(
            game_state_before_action, player_role, self.config, # Pass current game_state
            player_submitted_strategy, scherbius_encrypts_from_player,
            self.state.get("scherbius_planned_strategy"),
            self.state.get("scherbius_planned_encryption"),
            self._turing_ai_fn
        )

        final_turing_strategy = strategy_info["final_turing_strategy"]
        final_scherbius_strategy = strategy_info["final_scherbius_strategy"]
        scherbius_encryption_for_step = strategy_info["scherbius_encryption_for_step"]

        if player_role == "Turing" and final_scherbius_strategy is None:
             return None, "AI Scherbius plan missing. Please restart the game or report this bug."

        if player_role == "Scherbius":
            self.state["scherbius_planned_strategy"] = strategy_info["updated_scherbius_planned_strategy"]
            self.state["scherbius_planned_encryption"] = strategy_info["updated_scherbius_planned_encryption"]
            self.state["turing_planned_strategy"] = strategy_info["turing_planned_strategy_for_state"]
        
        prev_t_points = game_state_before_action.turing_points
        prev_s_points = game_state_before_action.scherbius_points

        # game_state_after_action is the state AFTER the round's actions are processed
        game_state_after_action, battle_outcomes = tvs.py_process_step(
            game_state_before_action, # Pass the state before action
            final_turing_strategy,
            final_scherbius_strategy,
            scherbius_encryption_for_step)

        current_t_points = game_state_after_action.turing_points
        current_s_points = game_state_after_action.scherbius_points
        
        # Set last_round_summary based on the action that just completed
        self.state["last_round_summary"] = utils.create_last_round_summary_data(
            prev_t_points, prev_s_points, current_t_points, current_s_points,
            final_turing_strategy, final_scherbius_strategy,
            scherbius_encryption_for_step, self.config.n_battles
        )

        round_number = len(self.state.get("round_history", [])) + 1
        
        # current_round_potential_rewards in self.state was set at the start of the turn that just ended.
        # This is appropriate for logging the rewards that *were* available for the completed round.
        historical_entry = utils.create_historical_round_entry_data(
            round_number, current_t_points, current_s_points,
            scherbius_encryption_for_step, battle_outcomes,
            final_turing_strategy, final_scherbius_strategy,
            self.state.get("current_round_potential_rewards"), 
            self.config.n_battles
        )
        self.state.setdefault("round_history", []).append(historical_entry)

        # Prepare client data for the START of the NEXT round/turn.
        # Base this on game_state_after_action.
        # is_new_round_for_player should always be True here because a game step has occurred,
        # and the hand display needs to be regenerated from the new ground truth.
        client_data = self.prepare_round_start_data(
            game_state_after_action, 
            is_new_round_for_player=True
        )
        
        # self.state['game_state'] is now updated inside prepare_round_start_data.
        return client_data, None