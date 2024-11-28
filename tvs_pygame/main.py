"""
Building a very simple game.

The game is called "Turing vs. Scherbius".

"""

import json
import pygame
import fire

from turing_vs_scherbius import PyGameState, PyGameConfig
from nts_player import NeuralTSPlayer

from ts_game.utils import render_turing, HumanTSPlayer

def main(config_path, load_path, player_name):
    """
    Args:
        config_path: str, path to the configuration file.
        load_path: str, path to the model to load.
        player_name: str, name of the player. (e.g. 'turing' or 'scherbius')
    """
    game = TuringVsSherbius(player_name, config_path, load_path)
    game.run()

if __name__ == "__main__":
    fire.Fire(main)