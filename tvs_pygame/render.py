import pygame
import os
from glob import glob
import re

VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']

def cardpath_to_name(cardpath):
    """
    Paths are of the form /path/to/card/5_of_hearts.png
    This function extracts the card name from the path.
    Returns; (5, 'hearts')
    """
    basename = os.path.splitext(os.path.basename(cardpath))[0]
    match = re.match(r'(\w+)_of_(\w+)', basename)
    v = match.group(1)
    s = match.group(2)
    assert v in VALUES, f"Value {v} not in {VALUES}"
    assert s in SUITS, f"Suit {s} not in {SUITS}"
    return v, s

class Render():
    # TODO move to svgs
    def __init__(self, config, screen, player_name, card_png_folder):
        self.config = config
        self.screen = screen
        self.player_name = player_name
        self.card_images = self.load_images(card_png_folder)
        self.font = pygame.font.SysFont(None, 24)  # Load a default font

    def load_images(self, card_png_folder):
        assert os.path.exists(card_png_folder), f"Path {card_png_folder} does not exist."
        card_images = {}
        card_paths = glob(f'{card_png_folder}/*.png')
        for card_path in card_paths:
            card_name = cardpath_to_name(card_path)
            card_images[card_name] = pygame.image.load(card_path)
        return card_images

    def draw_card(self, card_name, x, y):
        self.screen.blit(self.card_images[card_name], (x, y))

    def draw_hand(self, cards, y):
        """
        Draws a hand of cards at a specific vertical position y.
        """
        for i, card_name in enumerate(cards):
            self.draw_card(card_name, i * 100, y)

    def draw_rewards(self, rewards, y):
        """
        Draws rewards in the middle of the screen.
        Rewards are drawn horizontally staring at the given y position.
        """
        for i, reward in enumerate(rewards):
            x = i * 100
            if isinstance(reward, int):
                text = self.font.render(str(reward), True, (0, 0, 0))
                self.screen.blit(text, (x, y))
            else:
                for j, card_name in enumerate(reward):
                    self.draw_card(card_name, x, y + j * 100)

    def update(self, turing_hand, intercepted_scherbius_hand, scherbius_hand, rewards):
        """
        Calls the necessary drawing functions to render the current game state.
        """
        self.screen.fill((255, 255, 255))  # Fill the screen with white background

        intercepted_y = 100  # Fixed vertical position for intercepted cards
        rewards_y = 250  # Fixed vertical position for rewards
        hand_y = 400  # Fixed vertical position for player's hand

        # Depending on player's view, draw appropriate content
        if self.player_name == 'turing':
            self.draw_hand(turing_hand, hand_y)
            self.draw_hand(intercepted_scherbius_hand, intercepted_y)
        elif self.player_name == 'scherbius':
            self.draw_hand(scherbius_hand, hand_y)
        self.draw_rewards(rewards, rewards_y)

        pygame.display.flip()  # Update the display with new content
