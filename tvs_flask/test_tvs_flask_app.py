import pytest
from unittest.mock import MagicMock, patch, ANY
import json
import random # For the SUT's random usage

# Replace 'flask_game_app' with the actual name of your file if different
import app as sut

# --- Mock tvs_core components ---
class MockTVSConfig:
    def __init__(self, scherbius_starting=7, scherbius_deal=2,
                 turing_starting=5, turing_deal=1,
                 victory_points=100, n_battles=4,
                 encryption_cost=10, encryption_vocab_size=10,
                 encryption_k_rotors=1, verbose=False,
                 max_vp=10, max_draw=3,
                 max_cards_per_battle=3, max_hand_size=30):
        self.scherbius_starting = scherbius_starting
        self.scherbius_deal = scherbius_deal
        self.turing_starting = turing_starting
        self.turing_deal = turing_deal
        self.victory_points = victory_points
        self.n_battles = n_battles
        self.encryption_cost = encryption_cost
        self.encryption_vocab_size = encryption_vocab_size
        self.encryption_k_rotors = encryption_k_rotors
        self.verbose = verbose
        self.max_vp = max_vp
        self.max_draw = max_draw
        self.max_cards_per_battle = max_cards_per_battle
        self.max_hand_size = max_hand_size

class MockBattleOutcome:
    def __init__(self, turing_sum, scherbius_sum, turing_cards_won, turing_vp_won):
        self.turing_sum = turing_sum
        self.scherbius_sum = scherbius_sum
        self.turing_cards_won = turing_cards_won
        self.turing_vp_won = turing_vp_won

class MockTVSGameState:
    def __init__(self, config):
        self.config = config
        self._is_won = False
        self._winner = None
        self._turing_points = 0
        self._scherbius_points = 0
        self.scherbius_hand_mock_data = [10, 20, 30, 40, 50]
        self.turing_hand_mock_data = [1, 2, 3]
        self.intercepted_plays_mock_data = [[], [10], [], []] if config.n_battles == 4 else [[] for _ in range(config.n_battles)]
        self.rewards_mock_data = (
            [[c+100] for c in range(config.n_battles)], # card_rewards (e.g. [100], [101]...)
            [5 for _ in range(config.n_battles)]    # vp_rewards
        )
        self.battle_results_mock_data = [
            MockBattleOutcome(10, 5, [1], 5) for _ in range(config.n_battles)
        ]
        self.step_call_args = None
        self.scherbius_observation_called_count = 0
        self.turing_observation_called_count = 0

    def is_won(self): return self._is_won
    def winner(self): return self._winner
    def turing_points(self): return self._turing_points
    def scherbius_points(self): return self._scherbius_points
    
    def scherbius_observation(self):
        self.scherbius_observation_called_count += 1
        return list(self.scherbius_hand_mock_data) 

    def turing_observation(self, scherbius_plays_or_empty_list):
        self.turing_observation_called_count += 1
        if not any(scherbius_plays_or_empty_list): # AI Turing getting hand
             return list(self.turing_hand_mock_data), [[] for _ in range(self.config.n_battles)]
        return list(self.turing_hand_mock_data), list(self.intercepted_plays_mock_data)

    def rewards(self):
        return self.rewards_mock_data

    def step(self, turing_strategy, scherbius_strategy, scherbius_encrypts):
        self.step_call_args = (turing_strategy, scherbius_strategy, scherbius_encrypts)
        self._turing_points += 10 
        self._scherbius_points += 5

    def battle_results(self):
        return list(self.battle_results_mock_data)

# --- Pytest Fixtures ---

DEFINED_INITIAL_GAME_STATE = {
    "game_instance": None,
    "player_role": None,
    "scherbius_planned_strategy": None,
    "scherbius_planned_encryption": False,
    "turing_planned_strategy": None,
    "last_round_summary": None,
    "round_history": [],
    "current_round_potential_rewards": None,
    "last_client_data_prepared": None,
    "player_initial_hand_for_turn": []
}

@pytest.fixture(autouse=True)
def reset_global_game_state():
    """Ensures game_state is reset for each test using a deep copy."""
    sut.game_state.clear()
    # Use json loads/dumps for a simple deep copy of the initial state dict
    sut.game_state.update(json.loads(json.dumps(DEFINED_INITIAL_GAME_STATE)))

@pytest.fixture
def mock_tvs(monkeypatch):
    """Mocks the tvs_core module and re-initializes GAME_CONFIG in SUT."""
    mock_core_module = MagicMock()
    mock_core_module.PyGameConfig = MockTVSConfig
    mock_core_module.PyGameState = MockTVSGameState
    
    monkeypatch.setattr(sut, "tvs", mock_core_module)
    
    sut.GAME_CONFIG = MockTVSConfig(
        scherbius_starting=7, scherbius_deal=2, turing_starting=5, turing_deal=1,
        victory_points=100, n_battles=4, encryption_cost=10, encryption_vocab_size=10,
        encryption_k_rotors=1, verbose=False, max_vp=10, max_draw=3,
        max_cards_per_battle=3, max_hand_size=30
    )
    return mock_core_module

@pytest.fixture
def app(mock_tvs):
    """Provides the Flask app instance with testing config."""
    sut.app.config.update({"TESTING": True, "DEBUG": False})
    return sut.app

@pytest.fixture
def client(app):
    """Provides a test client for the Flask app."""
    return app.test_client()

@pytest.fixture
def mock_random_utilities(monkeypatch):
    """Mocks random functions for predictable AI behavior."""
    mock_shuffle = MagicMock()
    def shuffle_side_effect(x): x.reverse() 
    mock_shuffle.side_effect = shuffle_side_effect
    monkeypatch.setattr(sut.random, "shuffle", mock_shuffle)

    mock_random_val = MagicMock(return_value=0.1) # Ensures AI plays (thresholds 0.9, 0.7)
    monkeypatch.setattr(sut.random, "random", mock_random_val)

    mock_randint_val = MagicMock(return_value=1) # AI plays 1 card
    monkeypatch.setattr(sut.random, "randint", mock_randint_val)
    
    mock_choice_val = MagicMock(return_value=True) # Scherbius AI encrypts
    monkeypatch.setattr(sut.random, "choice", mock_choice_val)
    
    return {
        "shuffle": mock_shuffle, "random": mock_random_val,
        "randint": mock_randint_val, "choice": mock_choice_val
    }

def _common_ai_player_tests(player_func_output, hand, num_battles, game_config):
    strategy = player_func_output
    if isinstance(strategy, tuple): # Scherbius AI
        strategy, encrypt = strategy
        assert isinstance(encrypt, bool)

    assert isinstance(strategy, list)
    assert len(strategy) == num_battles
    played_cards_from_hand = []
    for battle_play in strategy:
        assert isinstance(battle_play, list)
        assert len(battle_play) <= game_config.max_cards_per_battle
        for card in battle_play:
            assert card in hand
            assert card not in played_cards_from_hand 
            played_cards_from_hand.append(card)
    return strategy


# --- Test AI Player Functions ---
def test_scherbius_ai_player(mock_random_utilities, mock_tvs):
    hand = [10, 20, 30, 40, 50]
    num_battles = sut.GAME_CONFIG.n_battles # Should be 4 from mock_tvs
    
    # Call the AI player function from the SUT
    strategy_tuple = sut.scherbius_ai_player(list(hand), num_battles) # Pass copy of hand
    
    _common_ai_player_tests(strategy_tuple, hand, num_battles, sut.GAME_CONFIG)
    strategy, encrypt = strategy_tuple
    assert encrypt is True # Due to mock_random_utilities.choice
    
    # Expected behavior with mocks:
    # hand_copy = [50,40,30,20,10] (after reverse)
    # Plays 1 card per battle (mock_randint_val = 1)
    # random.random() is 0.1, so plays if 0.1 < 0.9 (True)
    expected_played_cards_in_order = [10, 20, 30, 40] # Popped from reversed list
    for i in range(num_battles):
        if i < len(expected_played_cards_in_order):
             assert strategy[i] == [expected_played_cards_in_order[i]]
        else: # Not enough cards
             assert strategy[i] == []

def test_turing_ai_player(mock_random_utilities, mock_tvs):
    hand = [1, 2, 3, 4, 5]
    num_battles = sut.GAME_CONFIG.n_battles

    strategy = sut.turing_ai_player(list(hand), num_battles) # Pass copy of hand

    _common_ai_player_tests(strategy, hand, num_battles, sut.GAME_CONFIG)
    # Expected behavior with mocks:
    # hand_copy = [5,4,3,2,1] (after reverse)
    # Plays 1 card per battle (mock_randint_val = 1)
    # random.random() is 0.1, so plays if 0.1 < 0.7 (True)
    expected_played_cards_in_order = [1, 2, 3, 4] # Popped from reversed list
    for i in range(num_battles):
        if i < len(expected_played_cards_in_order):
            assert strategy[i] == [expected_played_cards_in_order[i]]
        else:
            assert strategy[i] == []


# --- Test prepare_round_start_data ---
def test_prepare_round_start_data_turing_role_new_round(mock_tvs, mock_random_utilities):
    mock_game = MockTVSGameState(sut.GAME_CONFIG)
    sut.game_state["game_instance"] = mock_game
    sut.game_state["player_role"] = "Turing"
    sut.game_state["last_round_summary"] = {"details": "old summary"}

    data = sut.prepare_round_start_data(is_new_round_for_player=True)

    assert data["player_role"] == "Turing"
    assert data["current_phase"] == "Turing_Action"
    assert len(data["player_hand"]) == len(mock_game.turing_hand_mock_data)
    assert data["player_hand"][0]["value"] == mock_game.turing_hand_mock_data[0]
    assert data["opponent_observed_plays"] == mock_game.intercepted_plays_mock_data
    assert sut.game_state["scherbius_planned_strategy"] is not None 
    assert sut.game_state["scherbius_planned_encryption"] is True # From mock_random_utilities
    assert data["scherbius_did_encrypt"] == sut.game_state["scherbius_planned_encryption"]
    assert data["last_round_summary"] is None 
    assert "rewards" in data
    assert not data["is_game_over"]

def test_prepare_round_start_data_scherbius_role_new_round(mock_tvs, mock_random_utilities):
    mock_game = MockTVSGameState(sut.GAME_CONFIG)
    sut.game_state["game_instance"] = mock_game
    sut.game_state["player_role"] = "Scherbius"
    
    data = sut.prepare_round_start_data(is_new_round_for_player=True)

    assert data["player_role"] == "Scherbius"
    assert data["current_phase"] == "Scherbius_Action"
    assert len(data["player_hand"]) == len(mock_game.scherbius_hand_mock_data)
    assert data["player_hand"][0]["value"] == mock_game.scherbius_hand_mock_data[0]
    assert data["opponent_observed_plays"] == [[] for _ in range(sut.GAME_CONFIG.n_battles)]
    assert data["scherbius_did_encrypt"] is False 
    assert sut.game_state["scherbius_planned_strategy"] is None 
    assert sut.game_state["scherbius_planned_encryption"] is False

def test_prepare_round_start_data_game_over(mock_tvs):
    mock_game = MockTVSGameState(sut.GAME_CONFIG)
    mock_game._is_won = True
    mock_game._winner = "Turing"
    sut.game_state["game_instance"] = mock_game
    sut.game_state["player_role"] = "Turing"

    data = sut.prepare_round_start_data(is_new_round_for_player=True) # is_new_round_for_player is effectively ignored if game over
    assert data["is_game_over"] is True
    assert data["winner"] == "Turing"
    assert data["current_phase"] == "Game_Over"

def test_prepare_round_start_data_preserves_summary_and_hand_if_not_new_round(mock_tvs, mock_random_utilities):
    mock_game = MockTVSGameState(sut.GAME_CONFIG)
    sut.game_state["game_instance"] = mock_game
    sut.game_state["player_role"] = "Turing"
    sut.game_state["last_round_summary"] = {"details": "preserved summary"}
    initial_hand_for_turn = [{"id": "pcard_0", "value": 99}]
    sut.game_state["player_initial_hand_for_turn"] = list(initial_hand_for_turn) # Store a copy
    
    data = sut.prepare_round_start_data(is_new_round_for_player=False)

    assert data["last_round_summary"] == {"details": "preserved summary"}
    assert data["player_hand"] == initial_hand_for_turn # Should use the pre-set hand


# --- Test Flask Endpoints ---

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    # Content check is minimal as it's a template
    assert b"<title>" in response.data # Basic check for HTML

@pytest.mark.parametrize("player_role", ["Turing", "Scherbius"])
def test_new_game_endpoint(client, player_role, mock_tvs, mock_random_utilities):
    response = client.post('/new_game', json={"player_role": player_role})
    assert response.status_code == 200
    data = response.json

    assert sut.game_state["player_role"] == player_role
    assert isinstance(sut.game_state["game_instance"], MockTVSGameState)
    assert sut.game_state["last_round_summary"] is None
    assert sut.game_state["round_history"] == []
    
    assert data["player_role"] == player_role
    assert not data["is_game_over"]
    if player_role == "Turing":
        assert data["current_phase"] == "Turing_Action"
        assert sut.game_state["scherbius_planned_strategy"] is not None
    else: # Scherbius
        assert data["current_phase"] == "Scherbius_Action"
        assert sut.game_state["scherbius_planned_strategy"] is None

def test_new_game_invalid_role(client):
    response = client.post('/new_game', json={"player_role": "InvalidRole"})
    assert response.status_code == 400
    data = response.json
    assert "Invalid player role" in data["error"]

def test_game_state_endpoint_not_started(client):
    sut.game_state["game_instance"] = None 
    response = client.get('/game_state')
    assert response.status_code == 404
    assert "Game not started" in response.json["error"]

def test_game_state_endpoint_with_cached_data(client, mock_tvs):
    cached_data = {"player_role": "Turing", "message": "cached data"}
    sut.game_state["game_instance"] = MockTVSGameState(sut.GAME_CONFIG) 
    sut.game_state["last_client_data_prepared"] = cached_data
    
    response = client.get('/game_state')
    assert response.status_code == 200
    assert response.json == cached_data

def test_game_state_endpoint_prepares_data_if_not_cached(client, mock_tvs, mock_random_utilities):
    mock_game = MockTVSGameState(sut.GAME_CONFIG)
    sut.game_state["game_instance"] = mock_game
    sut.game_state["player_role"] = "Turing"
    sut.game_state["last_client_data_prepared"] = None
    sut.game_state["last_round_summary"] = {"details": "existing summary"}

    response = client.get('/game_state')
    assert response.status_code == 200
    data = response.json
    assert data["player_role"] == "Turing"
    assert data["last_round_summary"] == {"details": "existing summary"} 

def test_game_state_endpoint_inconsistent_state(client, mock_tvs):
    sut.game_state["game_instance"] = MockTVSGameState(sut.GAME_CONFIG)
    sut.game_state["player_role"] = None 
    sut.game_state["last_client_data_prepared"] = None

    response = client.get('/game_state')
    assert response.status_code == 500
    assert "Game state is inconsistent" in response.json["error"]


# --- Test /submit_player_action endpoint ---

def _start_game_for_action(client, player_role):
    """Helper to start a game via the endpoint."""
    response = client.post('/new_game', json={"player_role": player_role})
    assert response.status_code == 200
    # mock_random_utilities fixture is active due to test function depending on it

def test_submit_action_turing_player(client, mock_tvs, mock_random_utilities):
    _start_game_for_action(client, "Turing")
    
    player_strategy = [[1], [2], [3], []] # Assumes hand [1,2,3] from MockTVSGameState
    
    action_payload = {"player_strategy": player_strategy}
    response = client.post('/submit_player_action', json=action_payload)
    assert response.status_code == 200
    data = response.json

    mock_game_instance = sut.game_state["game_instance"]
    assert mock_game_instance.step_call_args is not None
    submitted_t_strat, submitted_s_strat, submitted_s_encrypt = mock_game_instance.step_call_args
    
    assert submitted_t_strat == player_strategy
    assert submitted_s_strat == sut.game_state["scherbius_planned_strategy"] 
    assert submitted_s_encrypt == sut.game_state["scherbius_planned_encryption"]

    assert sut.game_state["last_round_summary"] is not None
    assert len(sut.game_state["round_history"]) == 1
    assert data["player_role"] == "Turing"
    assert data["current_phase"] == "Turing_Action" 
    assert data["last_round_summary"] is None # Cleared for new round display

def test_submit_action_scherbius_player(client, mock_tvs, mock_random_utilities):
    _start_game_for_action(client, "Scherbius")
    
    player_strategy = [[10], [20], [], [30]] # Assumes hand [10,20,30,40,50]
    scherbius_encrypts_choice = True
    
    action_payload = {
        "player_strategy": player_strategy,
        "scherbius_encrypts": scherbius_encrypts_choice
    }
    response = client.post('/submit_player_action', json=action_payload)
    assert response.status_code == 200
    data = response.json

    mock_game_instance = sut.game_state["game_instance"]
    assert mock_game_instance.step_call_args is not None
    submitted_t_strat, submitted_s_strat, submitted_s_encrypt = mock_game_instance.step_call_args
    
    assert submitted_s_strat == player_strategy
    assert submitted_s_encrypt == scherbius_encrypts_choice
    assert submitted_t_strat == sut.game_state["turing_planned_strategy"] 
    assert sut.game_state["turing_planned_strategy"] is not None 

    assert sut.game_state["last_round_summary"] is not None
    assert len(sut.game_state["round_history"]) == 1
    assert data["player_role"] == "Scherbius"
    assert data["current_phase"] == "Scherbius_Action"
    assert data["last_round_summary"] is None

def test_submit_action_game_over(client, mock_tvs):
    _start_game_for_action(client, "Turing")
    sut.game_state["game_instance"]._is_won = True 

    action_payload = {"player_strategy": [[] for _ in range(sut.GAME_CONFIG.n_battles)]}
    response = client.post('/submit_player_action', json=action_payload)
    assert response.status_code == 400
    assert "Game is over" in response.json["error"]

@pytest.mark.parametrize("invalid_strategy", [
    "not a list", # Wrong type
    [[]],         # Wrong length (not n_battles)
    [[1,2,3,4]]   # Too many cards in a battle (if max_cards_per_battle is 3)
    # Note: The SUT only checks list type and length of outer list.
    # Inner list length (max_cards_per_battle) is not validated by this endpoint directly,
    # but by the game logic (tvs_core) or AI player constraints.
    # The test for [[1,2,3,4]] might pass the endpoint validation if n_battles=1.
    # For this test, we focus on the endpoint's direct validations.
])
def test_submit_action_invalid_strategy_format(client, mock_tvs, invalid_strategy):
    _start_game_for_action(client, "Turing")
    
    action_payload = {"player_strategy": invalid_strategy}
    response = client.post('/submit_player_action', json=action_payload)
    assert response.status_code == 400
    assert "Invalid player strategy format" in response.json["error"]

def test_submit_action_turing_ai_scherbius_plan_missing(client, mock_tvs, mock_random_utilities):
    _start_game_for_action(client, "Turing") 
    sut.game_state["scherbius_planned_strategy"] = None # Manually corrupt state
    
    player_strategy = [[] for _ in range(sut.GAME_CONFIG.n_battles)]
    action_payload = {"player_strategy": player_strategy}
    response = client.post('/submit_player_action', json=action_payload)
    
    assert response.status_code == 500
    assert "AI Scherbius plan missing" in response.json["error"]

def test_submit_action_ends_game(client, mock_tvs, mock_random_utilities):
    _start_game_for_action(client, "Turing")
    
    mock_game_instance = sut.game_state["game_instance"]
    original_step_method = mock_game_instance.step
    def step_that_ends_game(*args, **kwargs):
        original_step_method(*args, **kwargs) # Call original mock step
        mock_game_instance._is_won = True
        mock_game_instance._winner = "Turing"
    # Patch the step method on the instance
    mock_game_instance.step = MagicMock(side_effect=step_that_ends_game)

    player_strategy = [[1], [2], [3], []] 
    action_payload = {"player_strategy": player_strategy}
    response = client.post('/submit_player_action', json=action_payload)
    assert response.status_code == 200
    data = response.json

    assert data["is_game_over"] is True
    assert data["winner"] == "Turing"
    assert data["current_phase"] == "Game_Over"
    # is_new_round_for_player was False, so last_round_summary should be the one just created
    assert data["last_round_summary"] is not None 
    assert data["last_round_summary"]["turing_points_gained_in_round"] == 10 # From mock step
    assert data["last_round_summary"]["scherbius_points_gained_in_round"] == 5