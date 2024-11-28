import fire
import pygame
import json

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
    fps = 1

    # Load the configuration
    with open("config.json", 'r') as f:
        config = json.loads(f.read())

    card_path = '/home/act65/Documents/playground/projects/turing-vs-scherbius/assets/PNG-cards-1.3/'

    render = Render(config, screen, 'turing', card_path)

    print(render.card_images)


    while True:
        clock.tick(fps)

        for card in [
            ('5', 'hearts'),
            ('6', 'hearts'),
            ('7', 'hearts'),
                     ]:
            print(card)
            render.draw_card(card, 10, 10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return



if __name__ == "__main__":
    fire.Fire(main)
