import json
import math
from turing_vs_scherbius import PyGameState, PyGameConfig

def random_action(turing_hand, scherbius_hand, rewards):
    return (
        [[turing_hand[0]]],
        [[scherbius_hand[0]]],
        [[]],
        False
    )

with open('config.json', 'r') as f:
    config = json.loads(f.read())

counter = 0
game_config = PyGameConfig(**config)
game_state = PyGameState(game_config)
while not game_state.is_won():
    counter += 1
    game_state.step(*random_action(game_state.turing_hand(), game_state.scherbius_hand(), game_state.rewards()))
    print(game_state.winner(), game_state.turing_points(), game_state.scherbius_points())
    print('rewards', game_state.rewards())
    print(counter)