import pygame

class HumanTSPlayer():
    def __init__(self, config, name):
        self.config = config
        self.name = name

    def select_action(self, hand, rewards, intercepted_hand=None):
        """
        Need to ask the user for input.

        Args:
            hand: list, the hand of the player.
            rewards: list, the rewards for the player.
            intercepted_hand: list, the intercepted hand of the other player.

        Returns:
            namedtuple, the action of the player.
                strategy: a list of list cards, the strategy of the player.
                
            if the player is Turing:
                guesses: a list cards, the guesses of the player.

            if the player is Scherbius:
                reencrypt: bool, whether to reencrypt the intercepted hand.
        """
        pass
