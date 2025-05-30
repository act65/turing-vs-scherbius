import os
from flask import Flask, render_template, jsonify, request
import random

import turing_vs_scherbius as tvs

app = Flask(__name__)

GAME_CONFIG = tvs.PyGameConfig(
    scherbius_starting=7, scherbius_deal=2,
    turing_starting=5, turing_deal=1,
    victory_points=100, n_battles=4,
    encryption_cost=10,
    encryption_code_len=1,
    encryption_vocab_size=10, verbose=False,
    max_vp=10, max_draw=3,        
    max_cards_per_battle=3, max_hand_size=30
)

game_state = {
    "game_instance": None,
    "scherbius_planned_strategy": None,
    "scherbius_planned_encryption": False,
    "last_round_summary": None,
    "turing_observed_scherbius_plays": None,
    "initial_turing_hand_for_turn": [],
    "round_history": [],
    "current_round_potential_rewards": None
}

def scherbius_ai_play(current_scherbius_hand, num_battles):
    strategy = [[] for _ in range(num_battles)]
    hand_copy = list(current_scherbius_hand)
    random.shuffle(hand_copy)
    for i in range(num_battles):
        if not hand_copy: break
        if random.random() < 0.6:
            num_cards_to_play = random.randint(1, min(2, len(hand_copy)))
            for _ in range(num_cards_to_play):
                if hand_copy:
                    strategy[i].append(hand_copy.pop())
    # Scherbius's decision to "encrypt" might still affect how its cards are shown,
    # even if Turing can't guess. Or this can be removed if not used.
    # For now, let's assume it might still visually obscure cards.
    encrypt = random.choice([True, False]) if GAME_CONFIG.encryption_code_len > 0 else False
    return strategy, encrypt

def prepare_round_start_data(is_new_round_for_turing=True):
    global game_state
    game = game_state["game_instance"]

    scherbius_raw_hand_values = game.scherbius_observation() 
    s_strategy, s_encrypts = scherbius_ai_play(scherbius_raw_hand_values, GAME_CONFIG.n_battles)
    
    game_state["scherbius_planned_strategy"] = s_strategy
    game_state["scherbius_planned_encryption"] = s_encrypts

    turing_hand_values, intercepted_plays_from_rust = game.turing_observation(s_strategy)
    
    if game_state["scherbius_planned_encryption"]:
        obscured_plays = []
        for battle_cards in intercepted_plays_from_rust:
            obscured_plays.append(['X' for _ in battle_cards])
        game_state["turing_observed_scherbius_plays"] = obscured_plays
    else:
        game_state["turing_observed_scherbius_plays"] = intercepted_plays_from_rust
    
    if is_new_round_for_turing:
        game_state["initial_turing_hand_for_turn"] = [
            {"id": f"tcard_{idx}", "value": val} for idx, val in enumerate(turing_hand_values)
        ]

    card_rewards, vp_rewards = game.rewards()
    # Store potential rewards for the round history
    game_state["current_round_potential_rewards"] = {"card_rewards": card_rewards, "vp_rewards": vp_rewards}


    client_data = {
        "turing_hand": game_state["initial_turing_hand_for_turn"],
        "scherbius_observed_plays": game_state["turing_observed_scherbius_plays"],
        "scherbius_did_encrypt": game_state["scherbius_planned_encryption"], # Use from game_state for consistency
        "rewards": {"card_rewards": card_rewards, "vp_rewards": vp_rewards},
        "turing_points": game.turing_points(),
        "max_victory_points": GAME_CONFIG.victory_points,
        "n_battles": GAME_CONFIG.n_battles,
        "max_cards_per_battle": GAME_CONFIG.max_cards_per_battle,
        "is_game_over": game.is_won(),
        "winner": game.winner() if game.is_won() else "Null",
        "last_round_summary": game_state["last_round_summary"],
        "current_phase": "Turing_Action",
        "round_history": game_state["round_history"]
    }
    if is_new_round_for_turing:
        game_state["last_round_summary"] = None
    return client_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    global game_state
    # Re-initialize config if encryption_code_len should be definitively zero
    current_config = tvs.PyGameConfig(
        scherbius_starting=GAME_CONFIG.scherbius_starting, scherbius_deal=GAME_CONFIG.scherbius_deal,
        turing_starting=GAME_CONFIG.turing_starting, turing_deal=GAME_CONFIG.turing_deal,
        victory_points=GAME_CONFIG.victory_points, n_battles=GAME_CONFIG.n_battles,
        encryption_cost=GAME_CONFIG.encryption_cost, encryption_code_len=0, # Force no encryption mechanic
        encryption_vocab_size=GAME_CONFIG.encryption_vocab_size, verbose=GAME_CONFIG.verbose,
        max_vp=GAME_CONFIG.max_vp, max_draw=GAME_CONFIG.max_draw,
        max_cards_per_battle=GAME_CONFIG.max_cards_per_battle, max_hand_size=GAME_CONFIG.max_hand_size
    )
    game_state["game_instance"] = tvs.PyGameState(current_config)
    game_state["last_round_summary"] = None
    game_state["round_history"] = []
    game_state["current_round_potential_rewards"] = None
    client_data = prepare_round_start_data(is_new_round_for_turing=True)
    return jsonify(client_data)

@app.route('/game_state', methods=['GET'])
def game_state_endpoint():
    if not game_state["game_instance"]:
        return jsonify({"error": "Game not started."}), 404
    
    game = game_state["game_instance"]
    card_rewards, vp_rewards = game.rewards()
    client_data = {
        "turing_hand": game_state["initial_turing_hand_for_turn"],
        "scherbius_observed_plays": game_state["turing_observed_scherbius_plays"],
        "scherbius_did_encrypt": game_state["scherbius_planned_encryption"],
        "rewards": {"card_rewards": card_rewards, "vp_rewards": vp_rewards },
        "turing_points": game.turing_points(),
        "max_victory_points": GAME_CONFIG.max_vp,
        "n_battles": GAME_CONFIG.n_battles,
        "is_game_over": game.is_won(),
        "winner": game.winner() if game.is_won() else "Null",
        "last_round_summary": game_state["last_round_summary"],
        "current_phase": "Turing_Action",
        "round_history": game_state["round_history"]
    }
    return jsonify(client_data)


@app.route('/submit_turing_action', methods=['POST'])
def submit_turing_action():
    global game_state
    game = game_state["game_instance"]
    if not game or game.is_won():
        return jsonify({"error": "Game is over or not initialized."}), 400

    data = request.json
    turing_submitted_strategy_values = data.get("turing_strategy")

    if not isinstance(turing_submitted_strategy_values, list) or \
       len(turing_submitted_strategy_values) != GAME_CONFIG.n_battles:
        return jsonify({"error": "Invalid Turing strategy format."}), 400

    scherbius_executed_strategy = game_state["scherbius_planned_strategy"]
    scherbius_executed_encryption = game_state["scherbius_planned_encryption"]

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
        "scherbius_encrypted_this_round": scherbius_executed_encryption,
        "battles": historical_battle_details
    }
    game_state["round_history"].append(historical_round_entry)
    
    client_data = prepare_round_start_data(is_new_round_for_turing=not game.is_won())
        
    return jsonify(client_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)