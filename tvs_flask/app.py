import os
from flask import Flask, render_template, jsonify, request
import random

import turing_vs_scherbius as tvs

app = Flask(__name__)

GAME_CONFIG = tvs.PyGameConfig(
    scherbius_starting=5, scherbius_deal=2,
    turing_starting=5, turing_deal=2,
    victory_points=10, n_battles=3,
    encryption_cost=0, encryption_code_len=2,
    encryption_vocab_size=10, verbose=False,
    max_vp=10, max_draw=3
)

game_state = {
    "game_instance": None,
    "scherbius_planned_strategy": None,
    "scherbius_planned_encryption": False,
    "last_round_summary": None,
    "turing_observed_scherbius_plays": None,
    "initial_turing_hand_for_turn": [] # Will store card objects {id, value}
}

def get_card_display_value(card_value): # This function might be redundant if we just use the value
    return card_value

def scherbius_ai_play(current_scherbius_hand, num_battles):
    strategy = [[] for _ in range(num_battles)]
    hand_copy = list(current_scherbius_hand) # Assuming current_scherbius_hand is just values
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

def prepare_round_start_data(is_new_round_for_turing=True):
    global game_state
    game = game_state["game_instance"]

    # Scherbius AI uses its raw hand (list of values)
    scherbius_raw_hand_values = game.scherbius_observation() 
    s_strategy, s_encrypts = scherbius_ai_play(scherbius_raw_hand_values, GAME_CONFIG.n_battles)
    
    game_state["scherbius_planned_strategy"] = s_strategy
    game_state["scherbius_planned_encryption"] = s_encrypts

    # Turing's observation gives their hand values and intercepted Scherbius plays
    turing_hand_values, intercepted_plays = game.turing_observation(s_strategy)
    game_state["turing_observed_scherbius_plays"] = intercepted_plays # This is Vec<Vec<u32>>
    
    if is_new_round_for_turing:
        # Assign temporary unique IDs to Turing's hand for client-side D&D
        game_state["initial_turing_hand_for_turn"] = [
            {"id": f"tcard_{idx}", "value": val} for idx, val in enumerate(turing_hand_values)
        ]

    card_rewards, vp_rewards = game.rewards()

    client_data = {
        "turing_hand": game_state["initial_turing_hand_for_turn"], # Now a list of {id, value} objects
        "scherbius_observed_plays": intercepted_plays, # Actual (maybe encrypted) card values
        "scherbius_did_encrypt": s_encrypts,
        "rewards": {"card_rewards": card_rewards, "vp_rewards": vp_rewards},
        "turing_points": game.turing_points(),
        "scherbius_points": game.scherbius_points(),
        "max_victory_points": GAME_CONFIG.max_vp,
        "n_battles": GAME_CONFIG.n_battles,
        "encryption_code_len": GAME_CONFIG.encryption_code_len,
        "is_game_over": game.is_won(),
        "winner": game.winner() if game.is_won() else "Null",
        "last_round_summary": game_state["last_round_summary"],
        "current_phase": "Turing_Action"
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
    game_state["game_instance"] = tvs.PyGameState(GAME_CONFIG)
    game_state["last_round_summary"] = None
    client_data = prepare_round_start_data(is_new_round_for_turing=True)
    return jsonify(client_data)

@app.route('/game_state', methods=['GET'])
def game_state_endpoint():
    if not game_state["game_instance"]:
        return jsonify({"error": "Game not started."}), 404
    
    game = game_state["game_instance"]
    card_rewards, vp_rewards = game.rewards()
    # Send the currently stored state for Turing, don't re-run AI
    client_data = {
        "turing_hand": game_state["initial_turing_hand_for_turn"],
        "scherbius_observed_plays": game_state["turing_observed_scherbius_plays"],
        "scherbius_did_encrypt": game_state["scherbius_planned_encryption"],
        "rewards": {"card_rewards": card_rewards, "vp_rewards": vp_rewards },
        "turing_points": game.turing_points(),
        "scherbius_points": game.scherbius_points(),
        "max_victory_points": GAME_CONFIG.max_vp,
        "n_battles": GAME_CONFIG.n_battles,
        "encryption_code_len": GAME_CONFIG.encryption_code_len,
        "is_game_over": game.is_won(),
        "winner": game.winner() if game.is_won() else "Null",
        "last_round_summary": game_state["last_round_summary"],
        "current_phase": "Turing_Action"
    }
    return jsonify(client_data)

@app.route('/submit_turing_action', methods=['POST'])
def submit_turing_action():
    global game_state
    game = game_state["game_instance"]
    if not game or game.is_won():
        return jsonify({"error": "Game is over or not initialized."}), 400

    data = request.json
    # Client sends strategy and guesses as arrays of card *values*
    turing_submitted_strategy_values = data.get("turing_strategy") 
    turing_submitted_guesses_values = data.get("turing_guesses")

    if not isinstance(turing_submitted_strategy_values, list) or \
       len(turing_submitted_strategy_values) != GAME_CONFIG.n_battles:
        return jsonify({"error": "Invalid Turing strategy format."}), 400
    # TODO: Add validation that submitted card values could have come from initial hand

    scherbius_executed_strategy = game_state["scherbius_planned_strategy"]
    scherbius_executed_encryption = game_state["scherbius_planned_encryption"]

    prev_t_points = game.turing_points()
    prev_s_points = game.scherbius_points()
    prev_encryption_broken_status = game.encryption_broken()

    game.step(turing_submitted_strategy_values, 
              scherbius_executed_strategy, 
              turing_submitted_guesses_values, 
              scherbius_executed_encryption)

    current_t_points = game.turing_points()
    current_s_points = game.scherbius_points()
    
    battle_details_summary = []
    for i in range(GAME_CONFIG.n_battles):
        battle_details_summary.append({
            "battle_id": i,
            "turing_played": turing_submitted_strategy_values[i], # Values submitted
            "scherbius_committed": scherbius_executed_strategy[i] 
        })

    game_state["last_round_summary"] = {
        "turing_points_gained_in_round": current_t_points - prev_t_points,
        "scherbius_points_gained_in_round": current_s_points - prev_s_points,
        "encryption_broken_this_round": game.encryption_broken() and not prev_encryption_broken_status,
        "encryption_attempted_by_turing": bool(turing_submitted_guesses_values and turing_submitted_guesses_values[0]),
        "battle_details": battle_details_summary,
        "scherbius_encrypted_last_round": scherbius_executed_encryption
    }
    
    client_data = prepare_round_start_data(is_new_round_for_turing=not game.is_won())
        
    return jsonify(client_data)

if __name__ == '__main__':
    app.run(debug=True)