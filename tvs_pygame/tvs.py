import json
import pygame
from turing_vs_scherbius import PyGameState, PyGameConfig
from nts_player import NeuralTSPlayer


from ts_game.render import Render
from ts_game.utils import HumanTSPlayer

class TuringVsSherbius():
    def __init__(self, player_name, config_path, load_path):
        # init pygame
        pygame.init()
        width, height = 800, 600
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Turing vs Sherbius')

        # Set up the frame rate
        self.clock = pygame.time.Clock()
        self.fps = 30

        # Load the configuration
        with open(config_path, 'r') as f:
            config = json.loads(f.read())

        self.config = PyGameConfig(**config)

        self.render = Render(screen, self.config.card_png_folder, player_name)

    def run(self):
        game_state = PyGameState(self.config)

        while not game_state.is_won():
            turing_hand, intercepted_scherbius_hand = game_state.turing_observation()
            scherbius_hand = game_state.scherbius_observation()
            rewards = game_state.rewards()
            
            # render current state
            self.render.update(turing_hand, intercepted_scherbius_hand, scherbius_hand, rewards)
            self.clock.tick(self.fps)

            # get actions
            turing_action = self.turing.select_action(turing_hand, rewards, intercepted_scherbius_hand)
            scherbius_action = self.scherbius.select_action(scherbius_hand, rewards)

            game_state.step(
                turing_action.strategy,
                scherbius_action.strategy,
                turing_action.guesses,
                scherbius_action.reencrypt
            )

        pygame.quit()