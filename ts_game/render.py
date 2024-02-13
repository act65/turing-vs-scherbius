import pygame
import os
from glob import glob

class Render():
    # TODO move to svgs
    def __init__(self, config, screen, player_name, card_png_folder):
        self.config = config
        self.screen = screen
        self.player_name = player_name
        self.card_images = self.load_images(card_png_folder)
        self.font = pygame.font.SysFont(None, 24)  # Load a default font

    def load_images(self, card_png_folder):
        card_images = {}
        card_paths = glob(f'{card_png_folder}/*.png')
        for card_path in card_paths:
            card_name = os.path.splitext(os.path.basename(card_path))[0]
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
