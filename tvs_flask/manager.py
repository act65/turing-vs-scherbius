import tvs_core as tvs # Assuming this is the actual library
from . import utils

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
        self.state["game_instance"] = tvs.PyGameState(self.config)
        self.state["player_role"] = player_role
        
        client_data = self.prepare_round_start_data(is_new_round_for_player=True)
        return client_data, None

    def get_current_game_state_for_client(self):
        if not self.state.get("game_instance"):
            return None, "Game not started. Please select a role to start a new game."

        if self.state.get("last_client_data_prepared"):
            return self.state["last_client_data_prepared"], None
        
        if self.state["player_role"]:
             client_data = self.prepare_round_start_data(is_new_round_for_player=False)
             return client_data, None
        return None, "Game state is inconsistent. Please start a new game."

    def prepare_round_start_data(self, is_new_round_for_player=True):
        game = self.state["game_instance"]
        player_role = self.state["player_role"]

        current_player_hand_values = []
        opponent_observed_plays = [[] for _ in range(self.config.n_battles)]
        scherbius_encryption_view = False
        current_phase = "Game_Over"

        if not game.is_won():
            round_start_info = utils.handle_pre_turn_observations_and_ai(
                game, player_role, self.config, self._scherbius_ai_fn
            )
            current_phase = round_start_info["current_phase"]
            current_player_hand_values = round_start_info["current_player_hand_values"]
            opponent_observed_plays = round_start_info["opponent_observed_plays_for_player"]
            scherbius_encryption_view = round_start_info["scherbius_encryption_status_for_view"]

            if player_role == "Turing":
                self.state["scherbius_planned_strategy"] = round_start_info["scherbius_planned_strategy"]
                self.state["scherbius_planned_encryption"] = round_start_info["scherbius_planned_encryption"]
        
        player_hand_client, new_initial_hand = utils.prepare_player_hand_for_display(
            current_player_hand_values,
            self.state.get("player_initial_hand_for_turn"),
            is_new_round_for_player
        )
        self.state["player_initial_hand_for_turn"] = new_initial_hand
        
        card_rewards, vp_rewards = game.rewards()
        self.state["current_round_potential_rewards"] = {"card_rewards": card_rewards, "vp_rewards": vp_rewards}

        client_data = utils.assemble_client_data_for_round_start(
            game, player_role, self.config, self.state,
            player_hand_client, opponent_observed_plays,
            scherbius_encryption_view, current_phase
        )

        if is_new_round_for_player and not game.is_won():
            self.state["last_round_summary"] = None

        self.state["last_client_data_prepared"] = client_data
        return client_data

    def submit_player_action(self, player_submitted_strategy, scherbius_encrypts_from_player=None):
        game = self.state["game_instance"]
        if not game or game.is_won():
            return None, "Game is over or not initialized."

        player_role = self.state["player_role"]

        if not isinstance(player_submitted_strategy, list) or \
           len(player_submitted_strategy) != self.config.n_battles:
            return None, "Invalid player strategy format."

        strategy_info = utils.determine_final_strategies(
            game, player_role, self.config,
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
        
        prev_t_points = game.turing_points()
        prev_s_points = game.scherbius_points()

        game.step(final_turing_strategy,
                  final_scherbius_strategy,
                  scherbius_encryption_for_step)

        current_t_points = game.turing_points()
        current_s_points = game.scherbius_points()
        
        self.state["last_round_summary"] = utils.create_last_round_summary_data(
            prev_t_points, prev_s_points, current_t_points, current_s_points,
            final_turing_strategy, final_scherbius_strategy,
            scherbius_encryption_for_step, self.config.n_battles
        )

        round_number = len(self.state.get("round_history", [])) + 1
        battle_outcomes = game.battle_results()
        
        historical_entry = utils.create_historical_round_entry_data(
            round_number, current_t_points, current_s_points,
            scherbius_encryption_for_step, battle_outcomes,
            final_turing_strategy, final_scherbius_strategy,
            self.state.get("current_round_potential_rewards"),
            self.config.n_battles
        )
        self.state.setdefault("round_history", []).append(historical_entry)

        client_data = self.prepare_round_start_data(is_new_round_for_player=not game.is_won())
        return client_data, None