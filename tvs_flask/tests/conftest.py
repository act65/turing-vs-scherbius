import pytest

# --- Mocks for testing ---
class MockGameConfig:
    def __init__(self):
        self.n_battles = 3
        self.max_cards_per_battle = 2 # Keep this small for easier testing
        self.victory_points = 10
        self.scherbius_starting = 5
        self.scherbius_deal = 2
        self.turing_starting = 5
        self.turing_deal = 2
        self.encryption_cost = 1
        self.encryption_vocab_size = 100
        self.encryption_k_rotors = 3
        self.max_vp = 5
        self.max_draw = 2
        self.max_hand_size = 10

@pytest.fixture
def mock_config():
    return MockGameConfig()

class MockBattleOutcome:
    def __init__(self, turing_sum, scherbius_sum, turing_cards_won=None, turing_vp_won=0):
        self.turing_sum = turing_sum
        self.scherbius_sum = scherbius_sum
        self.turing_cards_won = turing_cards_won if turing_cards_won is not None else []
        self.turing_vp_won = turing_vp_won

@pytest.fixture
def mock_battle_outcome_factory():
    return MockBattleOutcome


class MockPyGameState:
    def __init__(self, config):
        self.config = config
        self._turing_points = 0
        self._scherbius_points = 0
        self._is_won = False
        self._winner = "Null"
        self.scherbius_hand_cards = [1, 2, 3, 4, 5]
        self.turing_hand_cards = [10, 20, 30, 40, 50]
        self.current_card_rewards = [[11], [22], [33]]
        self.current_vp_rewards = [1, 2, 3]
        self.simulated_battle_results = [
            MockBattleOutcome(10, 5),
            MockBattleOutcome(3, 8),
            MockBattleOutcome(7, 7)
        ]
        self.step_called_with = None

    def scherbius_observation(self):
        return self.scherbius_hand_cards[:]

    def turing_observation(self, scherbius_strategy_for_interception):
        intercepted = [[] for _ in range(self.config.n_battles)]
        if scherbius_strategy_for_interception and \
           len(scherbius_strategy_for_interception) > 0 and \
           scherbius_strategy_for_interception[0]:
            intercepted[0] = [scherbius_strategy_for_interception[0][0]] # Simple mock
        return self.turing_hand_cards[:], intercepted

    def rewards(self):
        return self.current_card_rewards, self.current_vp_rewards

    def turing_points(self):
        return self._turing_points

    def scherbius_points(self):
        return self._scherbius_points

    def is_won(self):
        return self._is_won

    def winner(self):
        return self._winner

    def step(self, turing_strategy, scherbius_strategy, scherbius_encrypts):
        self.step_called_with = {
            "turing_strategy": turing_strategy,
            "scherbius_strategy": scherbius_strategy,
            "scherbius_encrypts": scherbius_encrypts
        }
        # Simulate point changes for testing history
        for i in range(self.config.n_battles):
            t_sum = sum(turing_strategy[i]) if turing_strategy[i] else 0
            s_sum = sum(scherbius_strategy[i]) if scherbius_strategy[i] else 0
            if t_sum > s_sum: self._turing_points += 1
            elif s_sum > t_sum: self._scherbius_points += 1
        
        if scherbius_encrypts:
            self._scherbius_points -= self.config.encryption_cost

        if self._turing_points >= self.config.victory_points:
            self._is_won = True; self._winner = "Turing"
        elif self._scherbius_points >= self.config.victory_points:
            self._is_won = True; self._winner = "Scherbius"


    def battle_results(self):
        # Return results based on the last step call if available
        if self.step_called_with:
            results = []
            for i in range(self.config.n_battles):
                t_sum = sum(self.step_called_with["turing_strategy"][i])
                s_sum = sum(self.step_called_with["scherbius_strategy"][i])
                results.append(MockBattleOutcome(t_sum, s_sum))
            return results
        return self.simulated_battle_results # Default before step

    # --- Test helpers for direct state manipulation ---
    def _set_won_state(self, is_won, winner, t_pts, s_pts):
        self._is_won = is_won
        self._winner = winner
        self._turing_points = t_pts
        self._scherbius_points = s_pts
    
    def _set_points(self, t_pts, s_pts):
        self._turing_points = t_pts
        self._scherbius_points = s_pts

@pytest.fixture
def mock_game_state_factory(mock_config):
    # This factory allows creating a new mock game state for each test if needed
    def _factory():
        return MockPyGameState(mock_config)
    return _factory

@pytest.fixture
def mock_game_state(mock_game_state_factory):
    return mock_game_state_factory()