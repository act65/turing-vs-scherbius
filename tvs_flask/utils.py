import random
import tvs_core as tvs

class GameManager:
    def __init__(self, game_config):
        self.config = game_config
        self.state = self._get_initial_state()

    def _get_initial_state(self):
        """Returns the initial structure for the game state."""
        return {
            "game_instance": None,
            "player_role": None,
            "scherbius_planned_strategy": None,
            "scherbius_planned_encryption": False,
            "turing_planned_strategy": None,
            "last_round_summary": None,
            "round_history": [],
            "current_round_potential_rewards": None,
            "last_client_data_prepared": None,
            "player_initial_hand_for_turn": [] # Stores card objects with IDs
        }

    def _scherbius_ai_player(self, hand, num_battles):
        """AI logic for Scherbius."""
        strategy = [[] for _ in range(num_battles)]
        hand_copy = list(hand)
        random.shuffle(hand_copy)
        for i in range(num_battles):
            if not hand_copy: break
            if random.random() < 0.9: # Threshold for playing cards
                num_cards_to_play = random.randint(1, min(self.config.max_cards_per_battle, len(hand_copy)))
                for _ in range(num_cards_to_play):
                    if hand_copy:
                        strategy[i].append(hand_copy.pop())
        encrypt = random.choice([True, False])
        return strategy, encrypt

    def _turing_ai_player(self, hand, num_battles):
        """AI logic for Turing."""
        strategy = [[] for _ in range(num_battles)]
        hand_copy = list(hand)
        random.shuffle(hand_copy)
        for i in range(num_battles):
            if not hand_copy: break
            if random.random() < 0.7: # Threshold for playing cards
                num_cards_to_play = random.randint(1, min(self.config.max_cards_per_battle, len(hand_copy)))
                for _ in range(num_cards_to_play):
                    if hand_copy:
                        strategy[i].append(hand_copy.pop())
        return strategy

    def new_game(self, player_role):
        """Starts a new game or resets the current one."""
        if player_role not in ["Turing", "Scherbius"]:
            return None, "Invalid player role specified."

        self.state = self._get_initial_state() # Reset state
        self.state["game_instance"] = tvs.PyGameState(self.config)
        self.state["player_role"] = player_role
        
        client_data = self.prepare_round_start_data(is_new_round_for_player=True)
        return client_data, None

    def get_current_game_state_for_client(self):
        """
        Prepares and returns the current game state for the client.
        Handles cases where the game might not be fully initialized.
        """
        if not self.state.get("game_instance"):
            return None, "Game not started. Please select a role to start a new game."

        if self.state.get("last_client_data_prepared"):
            return self.state["last_client_data_prepared"], None
        else:
            # This case implies a game instance exists, but no client data is cached.
            # This might happen if a client refreshes mid-game or there's an unusual flow.
            if self.state["player_role"]:
                 client_data = self.prepare_round_start_data(is_new_round_for_player=False)
                 return client_data, None
            return None, "Game state is inconsistent. Please start a new game."


    def prepare_round_start_data(self, is_new_round_for_player=True):
        """Prepares the data needed by the client at the start of a round or for display."""
        game = self.state["game_instance"]
        player_role = self.state["player_role"]

        current_player_hand_values = []
        opponent_observed_plays_for_player = [[] for _ in range(self.config.n_battles)]
        scherbius_encryption_status_for_view = False
        current_phase = "Game_Over"

        if not game.is_won():
            if player_role == "Turing":
                current_phase = "Turing_Action"
                scherbius_ai_hand = game.scherbius_observation()
                s_strategy, s_encrypts = self._scherbius_ai_player(scherbius_ai_hand, self.config.n_battles)
                self.state["scherbius_planned_strategy"] = s_strategy
                self.state["scherbius_planned_encryption"] = s_encrypts
                scherbius_encryption_status_for_view = s_encrypts

                turing_hand_val, intercepted_plays = game.turing_observation(s_strategy)
                current_player_hand_values = turing_hand_val
                opponent_observed_plays_for_player = intercepted_plays

            elif player_role == "Scherbius":
                current_phase = "Scherbius_Action"
                scherbius_hand_val = game.scherbius_observation()
                current_player_hand_values = scherbius_hand_val
                scherbius_encryption_status_for_view = False # Player Scherbius will decide
                # AI Turing plays after Scherbius submits, so no observed plays yet.

        player_hand_for_client = []
        if is_new_round_for_player or not self.state.get("player_initial_hand_for_turn"):
            self.state["player_initial_hand_for_turn"] = [
                {"id": f"pcard_{idx}", "value": val} for idx, val in enumerate(current_player_hand_values)
            ]
        player_hand_for_client = self.state["player_initial_hand_for_turn"]

        card_rewards, vp_rewards = game.rewards()
        self.state["current_round_potential_rewards"] = {"card_rewards": card_rewards, "vp_rewards": vp_rewards}

        client_data = {
            "player_role": player_role,
            "player_hand": player_hand_for_client,
            "opponent_observed_plays": opponent_observed_plays_for_player,
            "scherbius_did_encrypt": scherbius_encryption_status_for_view,
            "rewards": {"card_rewards": card_rewards, "vp_rewards": vp_rewards},
            "turing_points": game.turing_points(),
            "scherbius_points": game.scherbius_points(),
            "max_victory_points": self.config.victory_points,
            "n_battles": self.config.n_battles,
            "max_cards_per_battle": self.config.max_cards_per_battle,
            "is_game_over": game.is_won(),
            "winner": game.winner() if game.is_won() else "Null",
            "last_round_summary": self.state["last_round_summary"],
            "current_phase": current_phase,
            "round_history": self.state["round_history"],
            "scherbius_starting_cards": self.config.scherbius_starting,
            "scherbius_cards_deal_per_round": self.config.scherbius_deal,
            "turing_starting_cards": self.config.turing_starting,
            "turing_cards_deal_per_round": self.config.turing_deal,
            "encryption_cost": self.config.encryption_cost,
            "encryption_vocab_size": self.config.encryption_vocab_size,
            "encryption_k_rotors": self.config.encryption_k_rotors,
            "max_vp_reward_per_battle": self.config.max_vp,
            "max_card_reward_per_battle": self.config.max_draw,
            "max_hand_size": self.config.max_hand_size,
        }

        if is_new_round_for_player and not game.is_won():
            self.state["last_round_summary"] = None

        self.state["last_client_data_prepared"] = client_data
        return client_data

    def submit_player_action(self, player_submitted_strategy, scherbius_encrypts_from_player=None):
        """Processes the player's submitted action and advances the game."""
        game = self.state["game_instance"]
        if not game or game.is_won():
            return None, "Game is over or not initialized."

        player_role = self.state["player_role"]

        if not isinstance(player_submitted_strategy, list) or \
           len(player_submitted_strategy) != self.config.n_battles:
            return None, "Invalid player strategy format."

        final_turing_strategy = None
        final_scherbius_strategy = None
        scherbius_encryption_for_step = False

        if player_role == "Turing":
            final_turing_strategy = player_submitted_strategy
            final_scherbius_strategy = self.state["scherbius_planned_strategy"]
            scherbius_encryption_for_step = self.state["scherbius_planned_encryption"]
            if final_scherbius_strategy is None: # Should be planned in prepare_round_start
                 return None, "AI Scherbius plan missing. Please restart the game or report this bug."

        elif player_role == "Scherbius":
            final_scherbius_strategy = player_submitted_strategy
            scherbius_encryption_for_step = scherbius_encrypts_from_player if scherbius_encrypts_from_player is not None else False
            self.state["scherbius_planned_strategy"] = final_scherbius_strategy
            self.state["scherbius_planned_encryption"] = scherbius_encryption_for_step

            # AI Turing plans now, after Human Scherbius has committed
            # TODO: If AI Turing needs to see Scherbius's *encrypted* plays, this needs adjustment.
            # For now, turing_observation([]) just deals cards to AI Turing.
            turing_ai_hand, _ = game.turing_observation([]) # Deals cards to AI Turing
            final_turing_strategy = self._turing_ai_player(turing_ai_hand, self.config.n_battles)
            self.state["turing_planned_strategy"] = final_turing_strategy
        else:
            return None, "Invalid player role." # Should not happen if game started correctly

        prev_t_points = game.turing_points()
        prev_s_points = game.scherbius_points()

        game.step(final_turing_strategy,
                  final_scherbius_strategy,
                  scherbius_encryption_for_step)

        current_t_points = game.turing_points()
        current_s_points = game.scherbius_points()

        battle_outcomes = game.battle_results() # List of BattleOutcome objects
        
        battle_details_summary = []
        for i in range(self.config.n_battles):
            battle_details_summary.append({
                "battle_id": i,
                "turing_played": final_turing_strategy[i],
                "scherbius_committed": final_scherbius_strategy[i]
            })

        self.state["last_round_summary"] = {
            "turing_points_gained_in_round": current_t_points - prev_t_points,
            "scherbius_points_gained_in_round": current_s_points - prev_s_points,
            "battle_details": battle_details_summary,
            "scherbius_encrypted_last_round": scherbius_encryption_for_step
        }

        round_number = len(self.state["round_history"]) + 1
        historical_battle_details = []
        potential_rewards_for_round = self.state.get("current_round_potential_rewards", {
            "card_rewards": [[] for _ in range(self.config.n_battles)],
            "vp_rewards": [0] * self.config.n_battles
        })

        for i in range(self.config.n_battles):
            bo = battle_outcomes[i]
            battle_winner_text = "Draw"
            if bo.turing_sum > bo.scherbius_sum: battle_winner_text = "Turing"
            elif bo.scherbius_sum > bo.turing_sum: battle_winner_text = "Scherbius"

            historical_battle_details.append({
                "id": i,
                "turing_played_cards": final_turing_strategy[i],
                "scherbius_committed_cards": final_scherbius_strategy[i],
                "rewards_available_to_turing": { # These were the rewards *before* the battle
                    "vp": potential_rewards_for_round["vp_rewards"][i],
                    "cards": potential_rewards_for_round["card_rewards"][i]
                },
                "winner": battle_winner_text,
                # Could add actual rewards won here if needed from bo.turing_cards_won, bo.turing_vp_won
            })

        historical_round_entry = {
            "round_number": round_number,
            "turing_total_points_after_round": current_t_points,
            "scherbius_total_points_after_round": current_s_points,
            "scherbius_encrypted_this_round": scherbius_encryption_for_step,
            "battles": historical_battle_details
        }
        self.state["round_history"].append(historical_round_entry)

        client_data = self.prepare_round_start_data(is_new_round_for_player=not game.is_won())
        return client_data, None