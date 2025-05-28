import fire
import json

from nts_player import NTSPlayer
from rl_utils import ExperienceReplayBuffer

def main(config_path, buffer_address, load_dir):

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    game_config = PyGameConfig(**config)
    game_state = PyGameState(game_config)

    buffer = ExperienceReplayBuffer(buffer_address)

    turing_player = NTSPlayer(game_config, 'turing', load_dir)
    scherbius_player = NTSPlayer(game_config, 'scherbius', load_dir)

    while not game_state.is_won():
        turing_hand, intercepted_scherbius_hand = game_state.turing_observation()
        scherbius_hand = game_state.scherbius_observation()
        rewards = game_state.rewards()

        turing_action = turing_player.select_action(turing_hand, rewards, intercepted_scherbius_hand)
        scherbius_action = scherbius_player.select_action(scherbius_hand, rewards)

        game_state.step(
            turing_action.strategy,
            scherbius_action.strategy,
            turing_action.guesses,
            scherbius_action.reencrypt
        )

        buffer.add(
            turing_hand,
            intercepted_scherbius_hand,
            scherbius_hand,
            turing_action,
            scherbius_action,
            rewards,
        )

if __name__ == '__main__':
    fire.Fire(main)