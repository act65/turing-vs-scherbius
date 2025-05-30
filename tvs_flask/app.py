import os
from flask import Flask, render_template, jsonify, request
import random

import tvs_core as tvs

app = Flask(__name__)

GAME_CONFIG = tvs.PyGameConfig(
    scherbius_starting=7, scherbius_deal=2,
    turing_starting=5, turing_deal=1,
    victory_points=100, n_battles=4,
    encryption_cost=10,
    encryption_vocab_size=10, 
    encryption_k_rotors=1,
    verbose=False,
    max_vp=10, max_draw=3,        
    max_cards_per_battle=3, max_hand_size=30
)

game_state = {
    "game_instance": None,
    "player_role": "Turing",  # Added: Initialize to "Turing"
    "scherbius_planned_strategy": None,
    "scherbius_planned_encryption": False,
    "turing_planned_strategy": None,  # Added: For AI Turing
    "last_round_summary": None,
    "opponent_observed_plays_for_player": None,  # Renamed
    "initial_player_hand_for_turn": [],  # Renamed
    "round_history": [],
    "current_round_potential_rewards": None
}

def ai_player(hand, num_battles):
    strategy = [[] for _ in range(num_battles)]
    hand_copy = list(hand)
    random.shuffle(hand_copy)
    for i in range(num_battles):
        if not hand_copy: break
        if random.random() < 0.6:
            num_cards_to_play = random.randint(1, min(2, len(hand_copy)))
            for _ in range(num_cards_to_play):
                if hand_copy:
                    strategy[i].append(hand_copy.pop())

    encrypt = random.choice([True, False])
    return strategy, encrypt


def prepare_round_start_data(is_new_round_for_player=True): # Signature updated
    global game_state
    game = game_state["game_instance"]
    current_player_role = game_state["player_role"]

    client_data = {"player_role": current_player_role}
    player_true_hand_values = []
    observed_opponent_moves = []
    
    if current_player_role == "Turing":
        scherbius_raw_hand_values = game.scherbius_observation()
        s_strategy, s_encrypts = ai_player(scherbius_raw_hand_values, GAME_CONFIG.n_battles)

        game_state["scherbius_planned_strategy"] = s_strategy
        game_state["scherbius_planned_encryption"] = s_encrypts

        player_true_hand_values, observed_opponent_moves = game.turing_observation(s_strategy)
        game_state["opponent_observed_plays_for_player"] = observed_opponent_moves # Store for /game_state
        
        client_data["opponent_observed_plays"] = observed_opponent_moves
        client_data["opponent_did_encrypt"] = s_encrypts
        client_data["scherbius_points"] = "???" # Turing doesn't see Scherbius's points
        client_data["current_phase"] = "Turing_Action"

    elif current_player_role == "Scherbius":
        # AI plays as Turing
        turing_raw_hand_for_ai, _ = game.turing_observation([]) # AI Turing sees no Scherbius cards initially for its planning
        t_strategy, _ = ai_player(turing_raw_hand_for_ai, GAME_CONFIG.n_battles) # AI Turing encryption is ignored

        game_state["turing_planned_strategy"] = t_strategy
        # Scherbius makes decisions, so no "opponent_observed_plays_for_player" is stored in game_state from Scherbius's perspective of Turing's hand
        game_state["opponent_observed_plays_for_player"] = [] # Scherbius does not see AI Turing's planned/intercepted cards

        player_true_hand_values = game.scherbius_observation()

        client_data["opponent_observed_plays"] = []
        client_data["opponent_did_encrypt"] = False # AI Turing does not encrypt
        client_data["scherbius_points"] = game.scherbius_points() # Scherbius sees own points
        client_data["current_phase"] = "Scherbius_Action"

    # Common logic for both roles
    if is_new_round_for_player:
        game_state["initial_player_hand_for_turn"] = [
            {"id": f"card_{idx}", "value": val} for idx, val in enumerate(player_true_hand_values)
        ]

    client_data["player_hand"] = game_state["initial_player_hand_for_turn"]

    card_rewards, vp_rewards = game.rewards()
    game_state["current_round_potential_rewards"] = {"card_rewards": card_rewards, "vp_rewards": vp_rewards}

    client_data.update({
        "rewards": {"card_rewards": card_rewards, "vp_rewards": vp_rewards},
        "turing_points": game.turing_points(),
        "is_game_over": game.is_won(),
        "winner": game.winner() if game.is_won() else "Null",
        "last_round_summary": game_state["last_round_summary"],
        "round_history": game_state["round_history"],
        # Game Config Variables (ensure these are all GAME_CONFIG attributes)
        "config_scherbius_starting": GAME_CONFIG.scherbius_starting,
        "config_scherbius_deal": GAME_CONFIG.scherbius_deal,
        "config_turing_starting": GAME_CONFIG.turing_starting,
        "config_turing_deal": GAME_CONFIG.turing_deal,
        "config_victory_points": GAME_CONFIG.victory_points,
        "config_n_battles": GAME_CONFIG.n_battles,
        "config_encryption_cost": GAME_CONFIG.encryption_cost,
        "config_encryption_vocab_size": GAME_CONFIG.encryption_vocab_size,
        "config_encryption_k_rotors": GAME_CONFIG.encryption_k_rotors,
        "config_max_vp": GAME_CONFIG.max_vp,
        "config_max_draw": GAME_CONFIG.max_draw,
        "config_max_cards_per_battle": GAME_CONFIG.max_cards_per_battle,
        "config_max_hand_size": GAME_CONFIG.max_hand_size
    })
    # No need to pop old keys like "max_victory_points" as they are not added in the first place in this new structure

    if is_new_round_for_player:
        game_state["last_round_summary"] = None
    return client_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    global game_state
    data = request.json or {}
    player_role_from_request = data.get("player_role", "Turing")

    if player_role_from_request not in ["Turing", "Scherbius"]:
        player_role_from_request = "Turing" # Default to Turing if invalid role provided

    game_state["game_instance"] = tvs.PyGameState(GAME_CONFIG)
    game_state["player_role"] = player_role_from_request
    game_state["scherbius_planned_strategy"] = None
    game_state["scherbius_planned_encryption"] = False
    game_state["turing_planned_strategy"] = None
    game_state["last_round_summary"] = None
    game_state["opponent_observed_plays_for_player"] = None
    game_state["initial_player_hand_for_turn"] = []
    game_state["round_history"] = []
    game_state["current_round_potential_rewards"] = None

    # The parameter name for prepare_round_start_data will be changed in a later step
    # For now, assuming it's is_new_round_for_player as per instructions for that function
    client_data = prepare_round_start_data(is_new_round_for_player=True)
    return jsonify(client_data)

@app.route('/game_state', methods=['GET'])
def game_state_endpoint():
    if not game_state["game_instance"]:
        return jsonify({"error": "Game not started."}), 404
    
    game = game_state["game_instance"]
    card_rewards, vp_rewards = game.rewards() # These are general, not player-specific view of rewards

    current_player_role = game_state.get("player_role", "Turing")
    opponent_did_encrypt_value = False
    if current_player_role == "Turing":
        opponent_did_encrypt_value = game_state.get("scherbius_planned_encryption", False)
    # If player is Scherbius, AI Turing does not encrypt, so False is correct.

    scherbius_points_value = "???"
    if current_player_role == "Scherbius":
        scherbius_points_value = game.scherbius_points()

    current_phase_value = "Turing_Action" # Default
    if current_player_role == "Scherbius":
        current_phase_value = "Scherbius_Action"
    elif game.is_won(): # If game is over, phase might not be relevant or could be "GameOver"
        current_phase_value = "GameOver"


    client_data = {
        "player_role": current_player_role,
        "player_hand": game_state.get("initial_player_hand_for_turn", []),
        "opponent_observed_plays": game_state.get("opponent_observed_plays_for_player", []),
        "opponent_did_encrypt": opponent_did_encrypt_value,
        "rewards": {"card_rewards": card_rewards, "vp_rewards": vp_rewards },
        "turing_points": game.turing_points(),
        "scherbius_points": scherbius_points_value,
        "is_game_over": game.is_won(),
        "winner": game.winner() if game.is_won() else "Null",
        "last_round_summary": game_state["last_round_summary"],
        "current_phase": current_phase_value, # Updated based on role
        "round_history": game_state["round_history"],
        # Game Config Variables - Manually listed for safety and consistency
        "config_scherbius_starting": GAME_CONFIG.scherbius_starting,
        "config_scherbius_deal": GAME_CONFIG.scherbius_deal,
        "config_turing_starting": GAME_CONFIG.turing_starting,
        "config_turing_deal": GAME_CONFIG.turing_deal,
        "config_victory_points": GAME_CONFIG.victory_points,
        "config_n_battles": GAME_CONFIG.n_battles,
        "config_encryption_cost": GAME_CONFIG.encryption_cost,
        "config_encryption_vocab_size": GAME_CONFIG.encryption_vocab_size,
        "config_encryption_k_rotors": GAME_CONFIG.encryption_k_rotors,
        "config_max_vp": GAME_CONFIG.max_vp,
        "config_max_draw": GAME_CONFIG.max_draw,
        "config_max_cards_per_battle": GAME_CONFIG.max_cards_per_battle,
        "config_max_hand_size": GAME_CONFIG.max_hand_size
    }
    # No need to .pop() old keys as they are not added if using this direct construction.

    return jsonify(client_data)


@app.route('/submit_turing_action', methods=['POST'])
def submit_turing_action():
    global game_state
    game = game_state["game_instance"]
    if not game or game.is_won():
        return jsonify({"error": "Game is over or not initialized."}), 400

    if game_state.get("player_role", "Turing") != "Turing": # Added check
        return jsonify({"error": "Invalid action for current player role."}), 400

    data = request.json
    turing_submitted_strategy_values = data.get("turing_strategy")

    if not isinstance(turing_submitted_strategy_values, list) or \
       len(turing_submitted_strategy_values) != GAME_CONFIG.n_battles:
        return jsonify({"error": "Invalid Turing strategy format."}), 400

    # AI Scherbius makes decisions
    scherbius_executed_strategy = game_state["scherbius_planned_strategy"]
    scherbius_executed_encryption = game_state["scherbius_planned_encryption"]
    if scherbius_executed_strategy is None: # Should have been set in prepare_round_start_data
        return jsonify({"error": "AI Scherbius strategy not found."}), 500

    prev_t_points = game.turing_points()
    prev_s_points = game.scherbius_points()

    game.step(turing_submitted_strategy_values,
              scherbius_executed_strategy,
              scherbius_executed_encryption)

    current_t_points = game.turing_points()
    current_s_points = game.scherbius_points()
    
    # Get battle results to determine winners for history
    battle_outcomes = game.battle_results() # List of (t_sum, s_sum, t_cards_won, t_vp_won)
    battle_outcomes = [(bo.turing_sum, bo.scherbius_sum, bo.turing_cards_won, bo.turing_vp_won) for bo in battle_outcomes]

    # For last_round_summary (if still used by client for anything, though display is removed)
    battle_details_summary = []
    for i in range(GAME_CONFIG.n_battles):
        battle_details_summary.append({
            "battle_id": i,
            "turing_played": turing_submitted_strategy_values[i],
            "scherbius_committed": scherbius_executed_strategy[i]
            # Winner could be added here too if needed, but primary focus is history
        })

    game_state["last_round_summary"] = {
        "turing_points_gained_in_round": current_t_points - prev_t_points,
        "scherbius_points_gained_in_round": current_s_points - prev_s_points,
        "battle_details": battle_details_summary,
        "scherbius_encrypted_last_round": scherbius_executed_encryption
    }

    round_number = len(game_state["round_history"]) + 1
    
    historical_battle_details = []
    potential_rewards_for_round = game_state.get("current_round_potential_rewards", {
        "card_rewards": [[] for _ in range(GAME_CONFIG.n_battles)],
        "vp_rewards": [0] * GAME_CONFIG.n_battles
    })

    for i in range(GAME_CONFIG.n_battles):
        t_sum, s_sum, _, _ = battle_outcomes[i]
        battle_winner_text = "Draw"
        if t_sum > s_sum:
            battle_winner_text = "Turing"
        elif s_sum > t_sum:
            battle_winner_text = "Scherbius"

        historical_battle_details.append({
            "id": i,
            "turing_played_cards": turing_submitted_strategy_values[i],
            "scherbius_committed_cards": scherbius_executed_strategy[i],
            "rewards_available_to_turing": {
                "vp": potential_rewards_for_round["vp_rewards"][i],
                "cards": potential_rewards_for_round["card_rewards"][i]
            },
            "winner": battle_winner_text # ADDED WINNER
        })

    historical_round_entry = {
        "round_number": round_number,
        "turing_total_points_after_round": current_t_points,
        "scherbius_total_points_after_round": current_s_points,
        "scherbius_encrypted_this_round": scherbius_executed_encryption, # AI Scherbius's choice
        "battles": historical_battle_details
    }
    game_state["round_history"].append(historical_round_entry)

    client_data = prepare_round_start_data(is_new_round_for_player=not game.is_won())

    return jsonify(client_data)

@app.route('/submit_scherbius_action', methods=['POST'])
def submit_scherbius_action():
    global game_state
    game = game_state["game_instance"]

    if not game or game.is_won():
        return jsonify({"error": "Game is over or not initialized."}), 400

    if game_state.get("player_role") != "Scherbius":
        return jsonify({"error": "Invalid action for current player role."}), 400

    data = request.json
    scherbius_submitted_strategy_values = data.get("scherbius_strategy")
    scherbius_submitted_encryption = data.get("scherbius_encrypts", False) # Default to False if not provided

    if not isinstance(scherbius_submitted_strategy_values, list) or \
       len(scherbius_submitted_strategy_values) != GAME_CONFIG.n_battles:
        return jsonify({"error": "Invalid Scherbius strategy format."}), 400
    if not isinstance(scherbius_submitted_encryption, bool):
        return jsonify({"error": "Invalid Scherbius encryption format."}), 400

    turing_executed_strategy = game_state["turing_planned_strategy"]
    if not turing_executed_strategy: # Should have been set in prepare_round_start_data
        return jsonify({"error": "AI Turing strategy not found."}), 500


    prev_t_points = game.turing_points()
    prev_s_points = game.scherbius_points()

    game.step(turing_executed_strategy,
              scherbius_submitted_strategy_values,
              scherbius_submitted_encryption)

    current_t_points = game.turing_points()
    current_s_points = game.scherbius_points()

    battle_outcomes = [(bo.turing_sum, bo.scherbius_sum, bo.turing_cards_won, bo.turing_vp_won) for bo in game.battle_results()]

    battle_details_summary = []
    for i in range(GAME_CONFIG.n_battles):
        battle_details_summary.append({
            "battle_id": i,
            "turing_played": turing_executed_strategy[i],
            "scherbius_committed": scherbius_submitted_strategy_values[i]
        })

    game_state["last_round_summary"] = {
        "turing_points_gained_in_round": current_t_points - prev_t_points,
        "scherbius_points_gained_in_round": current_s_points - prev_s_points,
        "battle_details": battle_details_summary,
        "scherbius_encrypted_last_round": scherbius_submitted_encryption # From player's choice
    }

    round_number = len(game_state["round_history"]) + 1

    historical_battle_details = []
    potential_rewards_for_round = game_state.get("current_round_potential_rewards", {
        "card_rewards": [[] for _ in range(GAME_CONFIG.n_battles)],
        "vp_rewards": [0] * GAME_CONFIG.n_battles
    })

    for i in range(GAME_CONFIG.n_battles):
        t_sum, s_sum, _, _ = battle_outcomes[i]
        battle_winner_text = "Draw"
        if t_sum > s_sum:
            battle_winner_text = "Turing"
        elif s_sum > t_sum:
            battle_winner_text = "Scherbius"

        historical_battle_details.append({
            "id": i,
            "turing_played_cards": turing_executed_strategy[i], # AI Turing's play
            "scherbius_committed_cards": scherbius_submitted_strategy_values[i], # Player Scherbius's play
            "rewards_available_to_turing": { # Name remains, but context is "rewards of the battle"
                "vp": potential_rewards_for_round["vp_rewards"][i],
                "cards": potential_rewards_for_round["card_rewards"][i]
            },
            "winner": battle_winner_text
        })

    historical_round_entry = {
        "round_number": round_number,
        "turing_total_points_after_round": current_t_points,
        "scherbius_total_points_after_round": current_s_points,
        "scherbius_encrypted_this_round": scherbius_submitted_encryption, # Player Scherbius's choice
        "battles": historical_battle_details
    }
    game_state["round_history"].append(historical_round_entry)
    
    client_data = prepare_round_start_data(is_new_round_for_player=not game.is_won())
        
    return jsonify(client_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)