"""
Using cards from https://opengameart.org/content/playing-cards-vector-png
"""
import os
from glob import glob 
# TODO use svgs

import pygame

suits = ['spades', 'clubs', 'diamonds', 'hearts']
values = ['ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king']

def parse_name(card_name):
    """
    Args:
        card_name: str, the name of the card.
            6_of_spades.png, ... king_of_hearts.png, ...

    Returns:
        tuple, (int, int)
    """
    value, suit = card_name.split('_of_')
    return values.index(value) + 1, suits.index(suit)

def load_cards(path):
    """
    Args:
        path: str, path to the folder containing the card images.
            6_of_spades.png, ... king_of_hearts.png, ...

    Returns:
        dict
    """
    card_images = {}
    card_paths = glob(f'{path}/*.png')
    for card_path in card_paths:
        card_name = os.path.splitext(os.path.basename(card_path))[0]
        v, s = parse_name(card_name)
        card_images[(v, s)] = pygame.image.load(card_path)

    return card_images