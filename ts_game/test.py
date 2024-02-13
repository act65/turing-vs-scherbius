import fire
import pygame

from ts_game.render import Render

"""
A simple test of the rendering of the game.
"""

def main():
    # init pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Turing vs Sherbius')

    # Set up the frame rate
    clock = pygame.time.Clock()
    fps = 30

    # Load the configuration
    with open("config.json", 'r') as f:
        config = json.loads(f.read())

    render = Render(config, screen, 'turing', "projects/turing-vs-scherbius/ts_game/svgs")

    turing_hand = ['card_1', 'card_2', 'card_3', 'card_4', 'card_5']
    intercepted_scherbius_hand = ['card_1', 'card_2', 'card_3', 'card_4', 'card_5']
    scherbius_hand = ['card_1', 'card_2', 'card_3', 'card_4', 'card_5']
    rewards = [0, 0, 0, 0, 0]

    while True:
        render.update(turing_hand, intercepted_scherbius_hand, scherbius_hand, rewards)
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return



if __name__ == "__main__":
    fire.Fire(main)
