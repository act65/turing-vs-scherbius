import json
import pygame
from turing_vs_scherbius import PyGameState, PyGameConfig
from ts_game.utils import render_turing, HumanTSPlayer
from nts_player import NeuralTSPlayer

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

        # Create the players
        if player_name == 'turing':
            self.turing = HumanTSPlayer(config, 'turing')
            self.scherbius = NeuralTSPlayer(config, 'scherbius', load_path)
            self.render = self._render_turing
        elif player_name == 'scherbius':
            self.scherbius = HumanTSPlayer(config, 'scherbius')
            self.turing = NeuralTSPlayer(config, 'turing', load_path)
            self.render = self._render_scherbius
        else:
            raise ValueError('Invalid player name')

    def run(self):
        game_state = PyGameState(self.config)

        while not game_state.is_won():
            turing_hand, intercepted_scherbius_hand = game_state.turing_observation()
            scherbius_hand = game_state.scherbius_observation()
            rewards = game_state.rewards()
            
            # render current state
            self.render(turing_hand, intercepted_scherbius_hand, scherbius_hand, rewards)
            pygame.display.flip()
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

    def _render_turing(self, turing_hand, intercepted_scherbius_strat, scherbius_hand, rewards):
        """
        Turing gets to see their own hand 
        and the intercepted strategy of Scherbius.

        They also see the rewards.
        """
        pass

    def _render_scherbius(self, turing_hand, intercepted_scherbius_strat, scherbius_hand, rewards):
        """
        Scherbius gets to see their own hand.
        And the rewards.
        And the state of their enigma machine.
        """

    def _render_rewards(self, rewards):
        pass