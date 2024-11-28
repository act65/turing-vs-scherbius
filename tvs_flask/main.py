import os
from flask import Flask, send_file, render_template, session, redirect, url_for, request
import json
from turing_vs_scherbius import PyGameState, PyGameConfig
#, random_sherbius_player, validate_turing_action

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# for now, we will hardcode the user player
USER_PLAYER = 'turing'


####
# Card utilities
CARD_PATHS = os.listdir('assets/cards')

CARD_VALUES_MAP = {
    'a': 'ace',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '10': '10',
    'j': 'jack',
    'q': 'queen',
    'k': 'king'
}

CARD_SUITS_MAP = {
    'h': 'hearts',
    'd': 'diamonds',
    'c': 'clubs',
    's': 'spades'
}

def card_string_to_filename(card_string):
    """
    strings are in the format "{SUIT}-{VALUE}" where
        SUIT={"h", "d", "c", "s"} and VALUE={"a", "2", ..., "k"}
    want to reformat to "{LVALUE}_of_{LSUIT}.png" where
        LSUIT={"hearts", "diamonds", "clubs", "spades"} and
        LVALUE={"ace", "2", ..., "king"}
    """
    suit, value = card_string.split('-')
    suit = suit.lower()
    value = value.lower()
    suit = CARD_SUITS_MAP[suit]
    value = CARD_VALUES_MAP[value]
    return f"{value}_of_{suit}.png"
###

@app.route("/")
def index():
    return render_template("index.html",
        user_player=USER_PLAYER,
        turingCards=["h-2", "h-3", "h-8"],
        scherbiusCards=["h-5", "h-6", "h-7"],
        rewards=[["h-9", "h-10"], ["s-3"], ["h-j", "h-q"]]
    )

@app.route('/get_card_path')
def get_image():
    card = request.args.get('card_name')
    card = card_string_to_filename(card)
    if card in CARD_PATHS:
        image_path = os.path.join("assets/cards", card)
        print(f"Image path for card {card}: {image_path}")  # Log the image path
        return send_file(image_path, mimetype='image/png')
    else:
        return "Card not found", 404

@app.route('/get_vp_path')
def get_vp_image():
    vp = request.args.get('vp')
    card = card_string_to_filename(f's-{vp}')
    if card in CARD_PATHS:
        image_path = os.path.join("assets/cards", card)
        print(f"Image path for card {card}: {image_path}")
        return send_file(image_path, mimetype='image/png')
    else:
        return "Card not found", 404

if __name__ == "__main__":
    app.run()