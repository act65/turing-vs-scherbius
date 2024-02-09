import json
import fire

from nts_player import NTSPlayer
from rl_utils import ExperienceReplayBuffer

from turingscherbius import PyGameConfig

def main(config_path, epochs, batch_size, buffer_address, save_dir):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    game_config = PyGameConfig(**config)

    buffer = ExperienceReplayBuffer(buffer_address)

    turing_player = NTSPlayer(game_config, 'turing')
    scherbius_player = NTSPlayer(game_config, 'scherbius')

    for e in range(epochs):
        for batch in buffer.training_dataset(batch_size):
            turing_player.update(batch)
            scherbius_player.update(batch)

    turing_player.save(save_dir)
    scherbius_player.save(save_dir)
    

if __name__ == '__main__':
    fire.Fire(main)