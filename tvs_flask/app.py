from flask import Flask, render_template, jsonify, request
import random
import tvs_core as tvs
from manager import GameManager

# --- Game Configuration ---
GAME_CONFIG = tvs.GameConfig(
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

# --- Flask Application Setup ---
app = Flask(__name__)
game_manager = GameManager(GAME_CONFIG) # Instantiate the game manager

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_game', methods=['POST'])
def new_game_route():
    data = request.json
    player_role = data.get("player_role")

    client_data, error = game_manager.new_game(player_role)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(client_data)

@app.route('/game_state', methods=['GET'])
def game_state_endpoint():
    client_data, error = game_manager.get_current_game_state_for_client()
    if error:
        if "Game not started" in error:
            return jsonify({"error": error}), 404
        else: # "Inconsistent state" or other internal issues
            return jsonify({"error": error}), 500
    return jsonify(client_data)

@app.route('/submit_player_action', methods=['POST'])
def submit_player_action_route():
    data = request.json
    player_submitted_strategy = data.get("player_strategy")
    scherbius_encrypts = data.get("scherbius_encrypts") # Will be None if not provided (e.g. Turing player)

    client_data, error = game_manager.submit_player_action(
        player_submitted_strategy,
        scherbius_encrypts_from_player=scherbius_encrypts
    )

    if error:
        # Distinguish between client errors (400) and server errors (500)
        if "Invalid player strategy format" in error or "Game is over" in error:
            return jsonify({"error": error}), 400
        else: # e.g., "AI Scherbius plan missing", "Invalid player role" (internal logic error)
            return jsonify({"error": error}), 500
    return jsonify(client_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Added debug=True for development