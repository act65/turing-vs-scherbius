import os
from flask import Flask, render_template, jsonify, request
import random

import tvs_core as tvs

app = Flask(__name__)

GAME_CONFIG = tvs.PyGameConfig(
    scherbius_starting=7, 
    scherbius_deal=2,
    turing_starting=5, 
    turing_deal=1,
    victory_points=100, 
    n_battles=4,
    encryption_cost=10,
    encryption_vocab_size=10,
    encryption_k_rotors=1,
    verbose=False,
    max_vp=10, max_draw=3,
    max_cards_per_battle=3, 
    max_hand_size=30
)

game_state = {
    "game_instance": None,
    "player_role": None, # "Turing" or "Scherbius"
    "scherbius_planned_strategy": None, # Used if AI is Scherbius, or if player is Scherbius
    "scherbius_planned_encryption": False, # Used if AI is Scherbius, or if player is Scherbius
    "turing_planned_strategy": None, # Used if AI is Turing
    "last_round_summary": None,
    # "turing_observed_scherbius_plays" -> now part of client_data dynamically
    # "initial_turing_hand_for_turn" -> now part of client_data dynamically as "player_hand"
    "round_history": [],
    "current_round_potential_rewards": None,
    "last_client_data_prepared": None # Cache of the last state sent to client
}

def scherbius_ai_player(hand, num_battles): # Renamed from ai_player for clarity
    strategy = [[] for _ in range(num_battles)]
    hand_copy = list(hand)
    random.shuffle(hand_copy)
    for i in range(num_battles):
        if not hand_copy: break
        if random.random() < 0.9:
            num_cards_to_play = random.randint(1, min(GAME_CONFIG.max_cards_per_battle, len(hand_copy)))
            for _ in range(num_cards_to_play):
                if hand_copy:
                    strategy[i].append(hand_copy.pop())
    encrypt = random.choice([True, False])
    return strategy, encrypt

def turing_ai_player(hand, num_battles):
    strategy = [[] for _ in range(num_battles)]
    hand_copy = list(hand)
    random.shuffle(hand_copy)
    for i in range(num_battles):
        if not hand_copy: break
        if random.random() < 0.7:
            num_cards_to_play = random.randint(1, min(GAME_CONFIG.max_cards_per_battle, len(hand_copy)))
            for _ in range(num_cards_to_play):
                if hand_copy:
                    strategy[i].append(hand_copy.pop())
    return strategy

def prepare_round_start_data(is_new_round_for_player=True):
    global game_state
    game = game_state["game_instance"]
    player_role = game_state["player_role"]

    current_player_hand_values = []
    opponent_observed_plays_for_player = [[] for _ in range(GAME_CONFIG.n_battles)] # Default to empty
    scherbius_encryption_status_for_view = False # What the current player sees/sets for Scherbius's encryption
    current_phase = "Game_Over" # Default if game is won

    if not game.is_won():
        if player_role == "Turing":
            current_phase = "Turing_Action"
            # AI Scherbius plans
            scherbius_ai_hand = game.scherbius_observation() # Deals cards to AI Scherbius
            # Assuming scherbius_ai_player is defined elsewhere
            s_strategy, s_encrypts = scherbius_ai_player(scherbius_ai_hand, GAME_CONFIG.n_battles)
            game_state["scherbius_planned_strategy"] = s_strategy
            game_state["scherbius_planned_encryption"] = s_encrypts
            scherbius_encryption_status_for_view = s_encrypts

            # Human Turing observes and gets hand
            turing_hand_val, intercepted_plays = game.turing_observation(s_strategy) # Deals cards to Human Turing
            current_player_hand_values = turing_hand_val
            opponent_observed_plays_for_player = intercepted_plays

        elif player_role == "Scherbius":
            current_phase = "Scherbius_Action"
            # Human Scherbius gets hand
            scherbius_hand_val = game.scherbius_observation() # Deals cards to Human Scherbius
            current_player_hand_values = scherbius_hand_val
            # Scherbius player will decide encryption, so default is False for view
            scherbius_encryption_status_for_view = False
            # Opponent (AI Turing) plays are not observed at this stage by Scherbius
            opponent_observed_plays_for_player = [[] for _ in range(GAME_CONFIG.n_battles)]


    player_hand_for_client = []
    if is_new_round_for_player or not game_state.get("player_initial_hand_for_turn"): # Ensure hand is set up
         game_state["player_initial_hand_for_turn"] = [
            {"id": f"pcard_{idx}", "value": val} for idx, val in enumerate(current_player_hand_values)
        ]
    player_hand_for_client = game_state["player_initial_hand_for_turn"]


    card_rewards, vp_rewards = game.rewards()
    game_state["current_round_potential_rewards"] = {"card_rewards": card_rewards, "vp_rewards": vp_rewards}

    client_data = {
        "player_role": player_role,
        "player_hand": player_hand_for_client,
        "opponent_observed_plays": opponent_observed_plays_for_player,
        "scherbius_did_encrypt": scherbius_encryption_status_for_view,
        "rewards": {"card_rewards": card_rewards, "vp_rewards": vp_rewards},
        "turing_points": game.turing_points(),
        "scherbius_points": game.scherbius_points(),
        "max_victory_points": GAME_CONFIG.victory_points, # Already present
        "n_battles": GAME_CONFIG.n_battles, # Already present
        "max_cards_per_battle": GAME_CONFIG.max_cards_per_battle, # Already present
        "is_game_over": game.is_won(),
        "winner": game.winner() if game.is_won() else "Null",
        "last_round_summary": game_state["last_round_summary"],
        "current_phase": current_phase,
        "round_history": game_state["round_history"],
        "scherbius_starting_cards": GAME_CONFIG.scherbius_starting,
        "scherbius_cards_deal_per_round": GAME_CONFIG.scherbius_deal,
        "turing_starting_cards": GAME_CONFIG.turing_starting,
        "turing_cards_deal_per_round": GAME_CONFIG.turing_deal,
        "encryption_cost": GAME_CONFIG.encryption_cost,
        "encryption_vocab_size": GAME_CONFIG.encryption_vocab_size,
        "encryption_k_rotors": GAME_CONFIG.encryption_k_rotors,
        "max_vp_reward_per_battle": GAME_CONFIG.max_vp,
        "max_card_reward_per_battle": GAME_CONFIG.max_draw,
        "max_hand_size": GAME_CONFIG.max_hand_size,
    }

    if is_new_round_for_player and not game.is_won(): # Clear summary only if it's truly a new turn setup
        game_state["last_round_summary"] = None

    game_state["last_client_data_prepared"] = client_data # Cache this
    return client_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    global game_state
    data = request.json
    player_role = data.get("player_role")

    if player_role not in ["Turing", "Scherbius"]:
        return jsonify({"error": "Invalid player role specified."}), 400

    game_state["game_instance"] = tvs.PyGameState(GAME_CONFIG)
    game_state["player_role"] = player_role
    game_state["last_round_summary"] = None
    game_state["round_history"] = []
    game_state["current_round_potential_rewards"] = None
    game_state["scherbius_planned_strategy"] = None
    game_state["scherbius_planned_encryption"] = False
    game_state["turing_planned_strategy"] = None
    game_state["player_initial_hand_for_turn"] = [] # Reset this specifically

    client_data = prepare_round_start_data(is_new_round_for_player=True)
    return jsonify(client_data)

@app.route('/game_state', methods=['GET'])
def game_state_endpoint():
    global game_state
    if not game_state.get("game_instance"):
        # Attempt to gracefully handle if game not started but page is loaded
        # This could redirect to new game selection or show an error.
        # For now, return error, client should handle new game flow.
        return jsonify({"error": "Game not started. Please select a role to start a new game."}), 404
    
    if game_state.get("last_client_data_prepared"):
        return jsonify(game_state["last_client_data_prepared"])
    else:
        # This case should ideally be rare if new_game is always the entry point.
        # Force a preparation, but this might have side effects if not careful.
        # Better to rely on new_game setting this up.
        # For robustness, if a game instance exists but no client data, try to prep.
        # This assumes player_role is already set.
        if game_state["player_role"]:
             client_data = prepare_round_start_data(is_new_round_for_player=False) # False to not reset summary
             return jsonify(client_data)
        return jsonify({"error": "Game state is inconsistent. Please start a new game."}), 500


@app.route('/submit_player_action', methods=['POST'])
def submit_player_action():
    global game_state
    game = game_state["game_instance"]
    if not game or game.is_won():
        return jsonify({"error": "Game is over or not initialized."}), 400

    data = request.json
    player_role = game_state["player_role"]
    player_submitted_strategy = data.get("player_strategy")

    if not isinstance(player_submitted_strategy, list) or \
       len(player_submitted_strategy) != GAME_CONFIG.n_battles:
        return jsonify({"error": "Invalid player strategy format."}), 400

    final_turing_strategy = None
    final_scherbius_strategy = None
    scherbius_encryption_for_step = False

    if player_role == "Turing":
        final_turing_strategy = player_submitted_strategy
        # AI Scherbius's plan was made during prepare_round_start_data
        final_scherbius_strategy = game_state["scherbius_planned_strategy"]
        scherbius_encryption_for_step = game_state["scherbius_planned_encryption"]
        if final_scherbius_strategy is None: # Should not happen
             return jsonify({"error": "AI Scherbius plan missing."}), 500


    elif player_role == "Scherbius":
        final_scherbius_strategy = player_submitted_strategy
        scherbius_encryption_for_step = data.get("scherbius_encrypts", False)
        # Store player Scherbius's choices
        game_state["scherbius_planned_strategy"] = final_scherbius_strategy
        game_state["scherbius_planned_encryption"] = scherbius_encryption_for_step

        # AI Turing plans now
        turing_ai_hand, _ = game.turing_observation([]) # Deals cards to AI Turing
        final_turing_strategy = turing_ai_player(turing_ai_hand, GAME_CONFIG.n_battles)
        game_state["turing_planned_strategy"] = final_turing_strategy # Store for record if needed

    else:
        return jsonify({"error": "Invalid player role."}), 500

    prev_t_points = game.turing_points()
    prev_s_points = game.scherbius_points()

    game.step(final_turing_strategy,
              final_scherbius_strategy,
              scherbius_encryption_for_step)

    current_t_points = game.turing_points()
    current_s_points = game.scherbius_points()

    battle_outcomes = game.battle_results()
    battle_outcomes_tuples = [(bo.turing_sum, bo.scherbius_sum, bo.turing_cards_won, bo.turing_vp_won) for bo in battle_outcomes]

    battle_details_summary = []
    for i in range(GAME_CONFIG.n_battles):
        battle_details_summary.append({
            "battle_id": i,
            "turing_played": final_turing_strategy[i],
            "scherbius_committed": final_scherbius_strategy[i]
        })

    game_state["last_round_summary"] = {
        "turing_points_gained_in_round": current_t_points - prev_t_points,
        "scherbius_points_gained_in_round": current_s_points - prev_s_points,
        "battle_details": battle_details_summary,
        "scherbius_encrypted_last_round": scherbius_encryption_for_step
    }

    round_number = len(game_state["round_history"]) + 1
    historical_battle_details = []
    # Use potential rewards that were set at the start of this round
    potential_rewards_for_round = game_state.get("current_round_potential_rewards", {
        "card_rewards": [[] for _ in range(GAME_CONFIG.n_battles)],
        "vp_rewards": [0] * GAME_CONFIG.n_battles
    })

    for i in range(GAME_CONFIG.n_battles):
        t_sum, s_sum, _, _ = battle_outcomes_tuples[i]
        battle_winner_text = "Draw"
        if t_sum > s_sum: battle_winner_text = "Turing"
        elif s_sum > t_sum: battle_winner_text = "Scherbius"

        historical_battle_details.append({
            "id": i,
            "turing_played_cards": final_turing_strategy[i],
            "scherbius_committed_cards": final_scherbius_strategy[i],
            "rewards_available_to_turing": {
                "vp": potential_rewards_for_round["vp_rewards"][i],
                "cards": potential_rewards_for_round["card_rewards"][i]
            },
            "winner": battle_winner_text
        })

    historical_round_entry = {
        "round_number": round_number,
        "turing_total_points_after_round": current_t_points,
        "scherbius_total_points_after_round": current_s_points,
        "scherbius_encrypted_this_round": scherbius_encryption_for_step,
        "battles": historical_battle_details
    }
    game_state["round_history"].append(historical_round_entry)

    # Prepare data for the next turn or game over state
    # is_new_round_for_player should be true if game is not over, to reset hand display etc.
    client_data = prepare_round_start_data(is_new_round_for_player=not game.is_won())
        
    return jsonify(client_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)