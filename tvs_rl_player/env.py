import dm_env
from dm_env import specs
import numpy as np
import bsuite
from bsuite.environments import base
import turing_vs_scherbius as tvs # Your game library

# --- Constants agreed upon ---
MAX_HAND_SIZE = 30
MAX_CARDS_PER_BATTLE_STRATEGY = 3 # Max cards one player can commit to a single battle

class TvSEnvironment(): # Should inherit from bsuite.environments.base.Environment
    """
    A bsuite Environment wrapper for the Turing vs Scherbius game.
    """

    def __init__(self,
                 config: tvs.PyGameConfig,
                 opponent_policy: callable, # A function: obs -> action
                 player_perspective: str = "Turing", # "Turing" or "Scherbius"
                 seed: int = None):
        """
        Initializes the TvS Environment.

        Args:
            config: PyGameConfig object for the TvS game.
            player_perspective: "Turing" or "Scherbius", determining which player the agent controls.
            opponent_policy: A callable that takes an observation for the opponent
                             and returns an action for the opponent.
            seed: Optional random seed.
        """
        if player_perspective not in ["Turing", "Scherbius"]:
            raise ValueError("player_perspective must be 'Turing' or 'Scherbius'")

        self._config = config
        self._player_perspective = player_perspective
        self._opponent_policy = opponent_policy
        # self._rng = np.random.RandomState(seed) # For any internal stochasticity if needed

        # Initialize the game state.
        # This might be done in reset() for bsuite, but good to have an instance.
        self._game = tvs.PyGameState(self._config)
        self._n_battles = self._config.n_battles

        # Internal state for managing opponent's next move, especially for Turing
        self._current_scherbius_strategy_for_turing_obs = None
        self._current_scherbius_reencrypt_for_turing_obs = None

        # Define observation and action specs
        self._observation_spec = self._make_observation_spec()
        self._action_spec = self._make_action_spec()

    def _make_observation_spec(self):
        """Defines the observation structure for the agent."""
        obs_spec = {
            'my_hand': specs.BoundedArray(
                shape=(MAX_HAND_SIZE,),
                dtype=int, # Assuming cards are integers
                minimum=0, # 0 for no card/padding
                maximum=self._config.encryption_vocab_size, # Or a general card max value
                name='player_hand'
            ),
            'my_points': specs.Array(shape=(), dtype=int, name='player_points'),
            'opponent_points': specs.Array(shape=(), dtype=int, name='opponent_points'),
            'last_round_card_rewards': specs.BoundedArray(
                shape=(self._n_battles, MAX_CARDS_PER_BATTLE_STRATEGY), # Or just (self._n_battles, max_cards_won)
                dtype=int, minimum=0, maximum=self._config.encryption_vocab_size, # Or card max value
                name='card_rewards'
            ),
            'last_round_vp_rewards': specs.BoundedArray(
                shape=(self._n_battles,),
                dtype=int, minimum=-self._config.max_vp, maximum=self._config.max_vp, # Assuming VP rewards can be negative
                name='vp_rewards'
            )
        }
        if self._player_perspective == "Turing":
            obs_spec['intercepted_scherbius_strategy'] = specs.BoundedArray(
                shape=(self._n_battles, MAX_CARDS_PER_BATTLE_STRATEGY),
                dtype=int, minimum=0, maximum=self._config.encryption_vocab_size, # Encrypted cards
                name='intercepted_strategy'
            )
        return obs_spec

    def _make_action_spec(self):
        """Defines the action structure for the agent."""
        if self._player_perspective == "Turing":
            # Turing's action is just the strategy
            action_spec = specs.BoundedArray(
                shape=(self._n_battles, MAX_CARDS_PER_BATTLE_STRATEGY),
                dtype=int, minimum=0, maximum=self._config.encryption_vocab_size, # Card values
                name='turing_strategy'
            )
        else: # Scherbius
            action_spec = {
                'strategy': specs.BoundedArray(
                    shape=(self._n_battles, MAX_CARDS_PER_BATTLE_STRATEGY),
                    dtype=int, minimum=0, maximum=self._config.encryption_vocab_size,
                    name='scherbius_strategy'
                ),
                'reencrypt': specs.DiscreteArray(num_values=2, dtype=bool, name='reencrypt_choice')
            }
        return action_spec

    def _get_observation(self):
        """Constructs the current observation for the agent."""
        # This is a helper, actual observation construction will be more complex
        # and differ based on player_perspective.
        
        # Common elements
        my_hand_raw = self._game.turing_hand() if self._player_perspective == "Turing" else self._game.scherbius_hand()
        my_hand_padded = np.zeros(MAX_HAND_SIZE, dtype=int)
        my_hand_padded[:len(my_hand_raw)] = my_hand_raw

        my_points = self._game.turing_points() if self._player_perspective == "Turing" else self._game.scherbius_points()

        # Rewards from the *previous* step.
        # game.rewards() gives rewards for the round that just *finished*.
        # These need to be stored from the previous step's result.
        # For the first step (after reset), these would be neutral (e.g., zeros).
        # Let's assume self._last_cards_rewards and self._last_vp_rewards are stored.
        # last_cards_rewards_padded = ...
        # last_vp_rewards_padded = ...


        obs = {
            'my_hand': my_hand_padded,
            'my_points': my_points,
            'last_round_card_rewards': self._last_cards_rewards_padded, # Needs padding
            'last_round_vp_rewards': self._last_vp_rewards_padded,     # Needs padding
        }

        if self._player_perspective == "Turing":
            # Turing's observation includes Scherbius's *current* intended strategy
            # This _current_scherbius_strategy_for_turing_obs should have been
            # determined by the opponent_policy at the end of the previous step or reset.
            turing_hand_raw, intercepted_raw = self._game.turing_observation(
               self._current_scherbius_strategy_for_turing_obs
            )
            # (Re-fetch hand as turing_observation might be the canonical way)
            my_hand_padded[:len(turing_hand_raw)] = turing_hand_raw
            obs['my_hand'] = my_hand_padded
            obs['intercepted_scherbius_strategy'] = ... # Padded version of intercepted_raw
        else: # Scherbius
            # Scherbius's observation is simpler
            scherbius_hand_raw = self._game.scherbius_observation()
            my_hand_padded[:len(scherbius_hand_raw)] = scherbius_hand_raw
            obs['my_hand'] = my_hand_padded
        return obs

    def _prepare_opponent_turing_observation(self, scherbius_intended_strategy_for_current_round):
        """
        Helper to create the observation for an opponent Turing.
        This is called when the agent is Scherbius, and we need to get Turing's action.
        """
        turing_hand_raw, intercepted_raw = self._game.turing_observation(scherbius_intended_strategy_for_current_round)
        turing_hand_padded = np.zeros(MAX_HAND_SIZE, dtype=int)
        turing_hand_padded[:len(turing_hand_raw)] = turing_hand_raw

        intercepted_padded = ... # Pad intercepted_raw

        opponent_obs = {
            'my_hand': turing_hand_padded,
            'my_points': self._game.turing_points(),
            'last_round_card_rewards': self._last_cards_rewards_padded, # From perspective of Turing
            'last_round_vp_rewards': self._last_vp_rewards_padded,     # From perspective of Turing
            'intercepted_scherbius_strategy': intercepted_padded
        }
        return opponent_obs

    def _prepare_opponent_scherbius_observation(self):
        """
        Helper to create the observation for an opponent Scherbius.
        This is called when the agent is Turing, and we need Scherbius's next strategy
        for Turing's *next* observation.
        """
        scherbius_hand_raw = self._game.scherbius_observation()
        scherbius_hand_padded = np.zeros(MAX_HAND_SIZE, dtype=int)
        scherbius_hand_padded[:len(scherbius_hand_raw)] = scherbius_hand_raw
        
        opponent_obs = {
            'my_hand': scherbius_hand_padded,
            'my_points': self._game.scherbius_points(),
            'last_round_card_rewards': self._last_cards_rewards_padded, # From perspective of Scherbius
            'last_round_vp_rewards': self._last_vp_rewards_padded,     # From perspective of Scherbius
        }
        return opponent_obs

    # --- bsuite.Environment API methods ---

    def reset(self): # -> dm_env.TimeStep
        """Resets the environment to an initial state."""
        self._game = tvs.PyGameState(self._config) # Re-initialize the game

        # Reset internal reward trackers
        # self._last_cards_rewards_padded = np.zeros((self._n_battles, MAX_CARDS_PER_BATTLE_STRATEGY), dtype=int)
        # self._last_vp_rewards_padded = np.zeros(self._n_battles, dtype=int)
        self._last_cards_rewards_padded = [[0]*MAX_CARDS_PER_BATTLE_STRATEGY for _ in range(self._n_battles)]
        self._last_vp_rewards_padded = [0]*self._n_battles


        if self._player_perspective == "Turing":
            # Scherbius (opponent) needs to decide its first move for Turing's initial observation
            scherbius_obs_for_first_move = self._prepare_opponent_scherbius_observation()
            scherbius_action = self._opponent_policy(scherbius_obs_for_first_move)
            # Assuming scherbius_action is {'strategy': ..., 'reencrypt': ...}
            self._current_scherbius_strategy_for_turing_obs = scherbius_action['strategy']
            # self._current_scherbius_reencrypt_for_turing_obs = scherbius_action['reencrypt'] # Stored for the step

        current_observation = self._get_observation()
        # return dm_env.restart(current_observation)
        print("Env Reset. Initial observation generated.")
        # Simulating bsuite/dm_env TimeStep structure
        return {"observation": current_observation, "reward": None, "discount": None, "step_type": "restart"}

    def step(self, action): # -> dm_env.TimeStep
        """Applies an action, steps the game, and returns a new TimeStep."""
        if self._game.is_won():
            # Should not happen if reset is called correctly after termination.
            # return self.reset() # Or raise an error
            print("Error: Stepping a finished game. Resetting.")
            return self.reset()

        turing_action_strategy = None
        scherbius_action_strategy = None
        scherbius_action_reencrypt = False # Default

        initial_my_points = self._game.turing_points() if self._player_perspective == "Turing" else self._game.scherbius_points()

        if self._player_perspective == "Turing":
            turing_action_strategy = action # Agent's action
            # Scherbius's strategy was already determined (by opponent_policy)
            # and used for Turing's observation. It's stored in:
            # self._current_scherbius_strategy_for_turing_obs
            # self._current_scherbius_reencrypt_for_turing_obs
            scherbius_action_strategy = self._current_scherbius_strategy_for_turing_obs
            scherbius_action_reencrypt = self._current_scherbius_reencrypt_for_turing_obs # Make sure this is set in reset/previous step

        else: # Agent is Scherbius
            scherbius_action_strategy = action['strategy']
            scherbius_action_reencrypt = action['reencrypt']

            # Turing (opponent) needs to decide its move based on Scherbius's *current* play
            turing_obs_for_current_move = self._prepare_opponent_turing_observation(
                scherbius_intended_strategy_for_current_round=scherbius_action_strategy
            )
            turing_action_strategy = self._opponent_policy(turing_obs_for_current_move)

        # --- Execute the game step ---
        # Ensure strategies are in the correct list-of-lists format if not already
        # For example, if NN outputs flat arrays, reshape them here.
        # Assuming `action` and `opponent_policy` return them in game-compatible format.
        self._game.step(turing_action_strategy, scherbius_action_strategy, scherbius_action_reencrypt)

        # --- Post-step ---
        # Get rewards from the game (these are for the round that just completed)
        cards_rewards_raw, vp_rewards_raw = self._game.rewards()
        # Store them for the *next* observation (pad them appropriately)
        # self._last_cards_rewards_padded = ... pad(cards_rewards_raw) ...
        # self._last_vp_rewards_padded = ... pad(vp_rewards_raw) ...
        print(f"Game step executed. Raw rewards: VP={vp_rewards_raw}, Cards={cards_rewards_raw}")
        self._last_cards_rewards_padded = [r[:MAX_CARDS_PER_BATTLE_STRATEGY] + [0]*(MAX_CARDS_PER_BATTLE_STRATEGY-len(r)) if isinstance(r, list) else [0]*MAX_CARDS_PER_BATTLE_STRATEGY for r in cards_rewards_raw]
        self._last_vp_rewards_padded = list(vp_rewards_raw)


        # Calculate scalar reward for the agent
        current_my_points = self._game.turing_points() if self._player_perspective == "Turing" else self._game.scherbius_points()
        reward = float(current_my_points - initial_my_points)

        if self._game.is_won():
            observation = self._get_observation() # Get final observation
            # return dm_env.termination(reward=reward, observation=observation)
            print(f"Game Over. Winner: {self._game.winner()}. Final reward: {reward}")
            return {"observation": observation, "reward": reward, "discount": 0.0, "step_type": "termination"}
        else:
            # If game continues:
            # If agent is Turing, we need Scherbius's *next* move for Turing's *next* observation
            if self._player_perspective == "Turing":
                scherbius_obs_for_next_move = self._prepare_opponent_scherbius_observation()
                scherbius_next_action = self._opponent_policy(scherbius_obs_for_next_move)
                self._current_scherbius_strategy_for_turing_obs = scherbius_next_action['strategy']
                self._current_scherbius_reencrypt_for_turing_obs = scherbius_next_action['reencrypt']

            observation = self._get_observation()
            # return dm_env.transition(reward=reward, observation=observation, discount=1.0)
            print(f"Game continues. Reward: {reward}. Next observation generated.")
            return {"observation": observation, "reward": reward, "discount": 1.0, "step_type": "transition"}


    def observation_spec(self): # -> Specs
        """Returns the observation spec."""
        return self._observation_spec

    def action_spec(self): # -> Specs
        """Returns the action spec."""
        return self._action_spec

    def close(self):
        """Frees any resources."""
        # If the Rust game backend needs explicit cleanup, do it here.
        # For PyGameState, it might just be Python's GC.
        print("Environment closed.")
        pass

    # Potentially add bsuite specific methods like bsuite_info()
    def bsuite_info(self):
        return {} # Or any relevant metadata for bsuite logging