import random
import tvs_core

def get_initial_state():
    """Returns the initial structure for the game state."""
    return {
        "game_state": None,
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

def scherbius_ai_player_logic(hand, num_battles, max_cards_per_battle):
    """AI logic for Scherbius."""
    strategy = [[] for _ in range(num_battles)]
    hand_copy = list(hand) # Ensure we don't modify the original hand list
    random.shuffle(hand_copy)
    for i in range(num_battles):
        if not hand_copy: break
        # Threshold for playing cards (original: 0.9)
        if random.random() < 0.9: 
            num_cards_to_play = random.randint(1, min(max_cards_per_battle, len(hand_copy)))
            for _ in range(num_cards_to_play):
                if hand_copy:
                    strategy[i].append(hand_copy.pop())
    encrypt = random.choice([True, False])
    return strategy, encrypt

def turing_ai_player_logic(hand, num_battles, max_cards_per_battle):
    """AI logic for Turing."""
    strategy = [[] for _ in range(num_battles)]
    hand_copy = list(hand) # Ensure we don't modify the original hand list
    random.shuffle(hand_copy)
    for i in range(num_battles):
        if not hand_copy: break
        # Threshold for playing cards (original: 0.7)
        if random.random() < 0.7: 
            num_cards_to_play = random.randint(1, min(max_cards_per_battle, len(hand_copy)))
            for _ in range(num_cards_to_play):
                if hand_copy:
                    strategy[i].append(hand_copy.pop())
    return strategy

def handle_pre_turn_observations_and_ai(game_state, player_role, config, scherbius_ai_logic_fn):
    """
    Handles observations and AI planning at the start of a round.
    """
    current_phase = "Game_Over" 
    scherbius_planned_strategy = None
    scherbius_planned_encryption = False
    scherbius_encryption_status_for_view = False
    current_player_hand_values = []
    opponent_observed_plays_for_player = [[] for _ in range(config.n_battles)]

    if player_role == "Turing":
        current_phase = "Turing_Action"
        scherbius_ai_hand = game_state.scherbius_hand
        s_strategy, s_encrypts = scherbius_ai_logic_fn(
            scherbius_ai_hand, config.n_battles, config.max_cards_per_battle
        )
        scherbius_planned_strategy = s_strategy
        scherbius_planned_encryption = s_encrypts
        scherbius_encryption_status_for_view = s_encrypts

        game_state, intercepted_plays = tvs_core.py_intercept_scherbius_strategy(game_state, s_strategy)
        current_player_hand_values = game_state.turing_hand
        opponent_observed_plays_for_player = intercepted_plays

    elif player_role == "Scherbius":
        current_phase = "Scherbius_Action"
        scherbius_hand_val = game_state.scherbius_hand
        current_player_hand_values = scherbius_hand_val
        # scherbius_encryption_status_for_view is False as Player Scherbius decides

    return game_state, {
        "current_phase": current_phase,
        "scherbius_planned_strategy": scherbius_planned_strategy,
        "scherbius_planned_encryption": scherbius_planned_encryption,
        "scherbius_encryption_status_for_view": scherbius_encryption_status_for_view,
        "current_player_hand_values": current_player_hand_values,
        "opponent_observed_plays_for_player": opponent_observed_plays_for_player,
    }

def prepare_player_hand_for_display(current_player_hand_values, existing_player_initial_hand, is_new_round_for_player):
    """
    Prepares the player's hand with stable IDs for client display.
    Returns (hand_for_client, hand_to_store_in_state).
    """
    if is_new_round_for_player or not existing_player_initial_hand:
        # If it's a new round or no existing hand structure, create fresh items.
        new_initial_hand = [
            {"id": f"pcard_{idx}", "value": val} for idx, val in enumerate(current_player_hand_values) # Corrected typo
        ]
        return new_initial_hand, new_initial_hand
    # If not a new round and an existing structure is present, reuse it for client display consistency.
    # This is typically for refreshing the view without advancing the game turn.
    return existing_player_initial_hand, existing_player_initial_hand

def assemble_client_data_for_round_start(
    game_state, player_role, config, game_state_snapshot,
    player_hand_for_client, opponent_observed_plays,
    scherbius_did_encrypt_view, current_phase
):
    """
    Assembles the comprehensive client data dictionary.
    `game_state_snapshot` contains `last_round_summary`, `round_history`.
    """
    card_rewards, vp_rewards = game_state.rewards

    client_data = {
        "player_role": player_role,
        "player_hand": player_hand_for_client,
        "opponent_observed_plays": opponent_observed_plays,
        "scherbius_did_encrypt": scherbius_did_encrypt_view,
        "rewards": {"card_rewards": card_rewards, "vp_rewards": vp_rewards},
        "turing_points": game_state.turing_points,
        "scherbius_points": game_state.scherbius_points,
        "max_victory_points": config.victory_points,
        "n_battles": config.n_battles,
        "max_cards_per_battle": config.max_cards_per_battle,
        "is_game_over": game_state.is_won,
        "winner": game_state.winner if game_state.is_won else "Null",
        "last_round_summary": game_state_snapshot.get("last_round_summary"),
        "current_phase": current_phase,
        "round_history": game_state_snapshot.get("round_history", []),
        "scherbius_starting_cards": config.scherbius_starting,
        "scherbius_cards_deal_per_round": config.scherbius_deal,
        "turing_starting_cards": config.turing_starting,
        "turing_cards_deal_per_round": config.turing_deal,
        "encryption_cost": config.encryption_cost,
        "encryption_vocab_size": config.encryption_vocab_size,
        "encryption_k_rotors": config.encryption_k_rotors,
        "max_vp_reward_per_battle": config.max_vp,
        "max_card_reward_per_battle": config.max_draw,
        "max_hand_size": config.max_hand_size,
    }
    return client_data

def determine_final_strategies(
    game_state, player_role, config,
    player_submitted_strategy, scherbius_encrypts_from_player,
    current_scherbius_planned_strategy, current_scherbius_planned_encryption,
    turing_ai_logic_fn
):
    """
    Determines final strategies and AI Turing's plan if player is Scherbius.
    """
    final_turing_strategy = None
    final_scherbius_strategy = None
    scherbius_encryption_for_step = False
    
    updated_scherbius_planned_strategy = current_scherbius_planned_strategy
    updated_scherbius_planned_encryption = current_scherbius_planned_encryption
    turing_planned_strategy_for_state = None 

    if player_role == "Turing":
        final_turing_strategy = player_submitted_strategy
        final_scherbius_strategy = current_scherbius_planned_strategy
        scherbius_encryption_for_step = current_scherbius_planned_encryption
    elif player_role == "Scherbius":
        final_scherbius_strategy = player_submitted_strategy
        scherbius_encryption_for_step = bool(scherbius_encrypts_from_player)
        
        updated_scherbius_planned_strategy = final_scherbius_strategy
        updated_scherbius_planned_encryption = scherbius_encryption_for_step

        turing_ai_hand = game_state.turing_hand 
        final_turing_strategy = turing_ai_logic_fn(
            turing_ai_hand, config.n_battles, config.max_cards_per_battle
        )
        turing_planned_strategy_for_state = final_turing_strategy
    
    return {
        "final_turing_strategy": final_turing_strategy,
        "final_scherbius_strategy": final_scherbius_strategy,
        "scherbius_encryption_for_step": scherbius_encryption_for_step,
        "updated_scherbius_planned_strategy": updated_scherbius_planned_strategy,
        "updated_scherbius_planned_encryption": updated_scherbius_planned_encryption,
        "turing_planned_strategy_for_state": turing_planned_strategy_for_state,
    }

def create_last_round_summary_data(
    prev_t_points, prev_s_points, current_t_points, current_s_points,
    final_turing_strategy, final_scherbius_strategy,
    scherbius_encrypted_this_round, n_battles
):
    """Creates the summary data for the last completed round."""
    battle_details_summary = []
    for i in range(n_battles):
        battle_details_summary.append({
            "battle_id": i,
            "turing_played": final_turing_strategy[i] if final_turing_strategy else [],
            "scherbius_committed": final_scherbius_strategy[i] if final_scherbius_strategy else []
        })

    return {
        "turing_points_gained_in_round": current_t_points - prev_t_points,
        "scherbius_points_gained_in_round": current_s_points - prev_s_points,
        "battle_details": battle_details_summary,
        "scherbius_encrypted_last_round": scherbius_encrypted_this_round
    }

def create_historical_round_entry_data(
    round_number, current_t_points, current_s_points,
    scherbius_encrypted_this_round, battle_outcomes, 
    final_turing_strategy, final_scherbius_strategy,
    potential_rewards_for_round_state, 
    n_battles
):
    """Creates an entry for the round history."""
    historical_battle_details = []
    
    pot_rewards = potential_rewards_for_round_state
    if pot_rewards is None: # Default if not found in state (should be set)
        pot_rewards = {
            "card_rewards": [[] for _ in range(n_battles)],
            "vp_rewards": [0] * n_battles
        }

    for i in range(n_battles):
        bo = battle_outcomes[i] 
        battle_winner_text = "Draw"
        if bo.turing_sum > bo.scherbius_sum: battle_winner_text = "Turing"
        elif bo.scherbius_sum > bo.turing_sum: battle_winner_text = "Scherbius"

        historical_battle_details.append({
            "id": i,
            "turing_played_cards": final_turing_strategy[i] if final_turing_strategy else [],
            "scherbius_committed_cards": final_scherbius_strategy[i] if final_scherbius_strategy else [],
            "rewards_available_to_turing": { 
                "vp": pot_rewards["vp_rewards"][i],
                "cards": pot_rewards["card_rewards"][i]
            },
            "winner": battle_winner_text,
        })

    return {
        "round_number": round_number,
        "turing_total_points_after_round": current_t_points,
        "scherbius_total_points_after_round": current_s_points,
        "scherbius_encrypted_this_round": scherbius_encrypted_this_round,
        "battles": historical_battle_details
    }