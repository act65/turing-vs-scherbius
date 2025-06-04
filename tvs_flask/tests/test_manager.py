import pytest
from unittest.mock import patch, MagicMock, ANY
<<<<<<< HEAD
from ..manager import GameManager
from .. import utils # For get_initial_state
=======
from manager import GameManager
import utils # For get_initial_state
>>>>>>> feat/pure-intercept-scherbius
from .conftest import MockGameConfig, MockPyGameState # Import mocks from conftest

# This ensures that when GameManager imports tvs_core, it gets our mock
@pytest.fixture(autouse=True)
def mock_tvs_core(monkeypatch):
    mock_core = MagicMock()
    mock_core.PyGameState = MockPyGameState # Use the class from conftest
    monkeypatch.setitem(__import__('sys').modules, 'tvs_core', mock_core)
    return mock_core


@pytest.fixture
def game_config(mock_config): # Use the one from conftest
    return mock_config

@pytest.fixture
def manager(game_config):
    # Patch the AI functions within the GameManager instance for more direct control if needed,
    # or rely on the utils mocks if those are sufficient.
    # For now, let's assume utils.py is tested and its AI functions work as expected.
    gm = GameManager(game_config)
    # We can mock the AI functions it uses if we want to control their output directly in GM tests
    gm._scherbius_ai_fn = MagicMock(return_value=([[1],[1]], False)) # strat, encrypt
    gm._turing_ai_fn = MagicMock(return_value=([[2],[2]])) # strat
    return gm

def test_game_manager_init(manager, game_config):
    assert manager.config == game_config
    assert manager.state == utils.get_initial_state()
    assert callable(manager._scherbius_ai_fn)
    assert callable(manager._turing_ai_fn)

def test_new_game_invalid_role(manager):
    client_data, error = manager.new_game("InvalidRole")
    assert client_data is None
    assert error == "Invalid player role specified."

def test_new_game_valid_role(manager, game_config, mock_tvs_core):
    # Test for Turing
    with patch.object(manager, 'prepare_round_start_data', return_value="prepared_data_turing") as mock_prepare:
        client_data, error = manager.new_game("Turing")
        assert error is None
        assert client_data == "prepared_data_turing"
        assert isinstance(manager.state["game_state"], mock_tvs_core.PyGameState)
        assert manager.state["player_role"] == "Turing"
        mock_prepare.assert_called_once_with(is_new_round_for_player=True)
        # Check PyGameState was instantiated with the config
        mock_tvs_core.PyGameState.assert_called_with(game_config)


    # Reset for Scherbius test (or use a new manager instance)
    manager.state = utils.get_initial_state() # Manual reset for simplicity
    mock_tvs_core.PyGameState.reset_mock()

    with patch.object(manager, 'prepare_round_start_data', return_value="prepared_data_scherbius") as mock_prepare:
        client_data, error = manager.new_game("Scherbius")
        assert error is None
        assert client_data == "prepared_data_scherbius"
        assert isinstance(manager.state["game_state"], mock_tvs_core.PyGameState)
        assert manager.state["player_role"] == "Scherbius"
        mock_prepare.assert_called_once_with(is_new_round_for_player=True)
        mock_tvs_core.PyGameState.assert_called_with(game_config)


def test_get_current_game_state_for_client(manager, mock_game_state_factory):
    # Case 1: Game not started
    client_data, error = manager.get_current_game_state_for_client()
    assert client_data is None
    assert error == "Game not started. Please select a role to start a new game."

    # Case 2: Game started, last_client_data_prepared exists
    manager.state["game_state"] = mock_game_state_factory()
    manager.state["player_role"] = "Turing"
    manager.state["last_client_data_prepared"] = {"data": "cached"}
    client_data, error = manager.get_current_game_state_for_client()
    assert error is None
    assert client_data == {"data": "cached"}

    # Case 3: Game started, no last_client_data_prepared (should call prepare_round_start_data)
    manager.state["last_client_data_prepared"] = None
    with patch.object(manager, 'prepare_round_start_data', return_value={"data": "freshly_prepared"}) as mock_prepare:
        client_data, error = manager.get_current_game_state_for_client()
        assert error is None
        assert client_data == {"data": "freshly_prepared"}
        mock_prepare.assert_called_once_with(is_new_round_for_player=False)
    
    # Case 4: Inconsistent state (game_state exists, but no player_role)
    manager.state["player_role"] = None
    manager.state["last_client_data_prepared"] = None # Ensure prepare is not hit due to cache
    client_data, error = manager.get_current_game_state_for_client()
    assert client_data is None
    assert error == "Game state is inconsistent. Please start a new game."


def test_prepare_round_start_data_turing_player(manager, game_config, mock_game_state):
    manager.state["game_state"] = mock_game_state
    manager.state["player_role"] = "Turing"
    manager.state["player_initial_hand_for_turn"] = [] # Ensure it's fresh

    # Mock the AI function used internally by handle_pre_turn_observations_and_ai
    # This AI function is part of utils, so we patch it there if we want to control its behavior
    # For this test, we are testing GameManager's integration, so we can rely on the manager's _scherbius_ai_fn mock
    # which is already set up in the `manager` fixture.
    # manager._scherbius_ai_fn = MagicMock(return_value=([[7],[8]], True)) # strat, encrypt

    client_data = manager.prepare_round_start_data(is_new_round_for_player=True)

    manager._scherbius_ai_fn.assert_called_once_with(
        mock_game_state.scherbius_hand_cards, game_config.n_battles, game_config.max_cards_per_battle
    )
    assert manager.state["scherbius_planned_strategy"] == [[1],[1]] # From manager fixture mock
    assert manager.state["scherbius_planned_encryption"] is False # From manager fixture mock
    assert client_data["player_role"] == "Turing"
    assert client_data["current_phase"] == "Turing_Action"
    assert len(client_data["player_hand"]) == len(mock_game_state.turing_hand_cards)
    assert client_data["player_hand"][0]["value"] == mock_game_state.turing_hand_cards[0]
    assert manager.state["last_round_summary"] is None # is_new_round_for_player=True

def test_prepare_round_start_data_scherbius_player(manager, game_config, mock_game_state):
    manager.state["game_state"] = mock_game_state
    manager.state["player_role"] = "Scherbius"
    manager.state["player_initial_hand_for_turn"] = []

    # Reset mock for this specific test path if it was called in a previous one by another role
    manager._scherbius_ai_fn.reset_mock() 

    client_data = manager.prepare_round_start_data(is_new_round_for_player=True)

    manager._scherbius_ai_fn.assert_not_called() # Player is Scherbius, AI Scherbius doesn't plan for itself
    assert manager.state.get("scherbius_planned_strategy") is None # Not set by this path
    assert client_data["player_role"] == "Scherbius"
    assert client_data["current_phase"] == "Scherbius_Action"
    assert len(client_data["player_hand"]) == len(mock_game_state.scherbius_hand_cards)
    assert client_data["player_hand"][0]["value"] == mock_game_state.scherbius_hand_cards[0]

def test_prepare_round_start_data_game_over(manager, mock_game_state):
    manager.state["game_state"] = mock_game_state
    manager.state["player_role"] = "Turing"
    mock_game_state._set_won_state(True, "Turing", 10, 5)

    client_data = manager.prepare_round_start_data(is_new_round_for_player=False) # Game over, so not new round for player
    
    assert client_data["is_game_over"] is True
    assert client_data["winner"] == "Turing"
    assert client_data["current_phase"] == "Game_Over"
    assert manager.state["last_round_summary"] is not None # Should not be cleared if game is over


def test_submit_player_action_game_over_or_not_init(manager, mock_game_state):
    # Game not initialized
    _, error = manager.submit_player_action([[]]*manager.config.n_battles)
    assert error == "Game is over or not initialized."

    # Game over
    manager.state["game_state"] = mock_game_state
    mock_game_state._set_won_state(True, "Turing", 10, 0)
    _, error = manager.submit_player_action([[]]*manager.config.n_battles)
    assert error == "Game is over or not initialized."

def test_submit_player_action_invalid_strategy_format(manager, mock_game_state):
    manager.state["game_state"] = mock_game_state
    manager.state["player_role"] = "Turing"
    mock_game_state._set_won_state(False, "Null", 0, 0) # Game ongoing

    _, error = manager.submit_player_action("not a list")
    assert error == "Invalid player strategy format."

    _, error = manager.submit_player_action([[]]*(manager.config.n_battles - 1)) # Wrong number of battles
    assert error == "Invalid player strategy format."

def test_submit_player_action_turing_player(manager, game_config, mock_game_state):
    manager.state["game_state"] = mock_game_state
    manager.state["player_role"] = "Turing"
    # AI Scherbius plan needs to be in state (set by prepare_round_start_data)
    manager.state["scherbius_planned_strategy"] = [[1],[1],[1]] # Mocked AI plan
    manager.state["scherbius_planned_encryption"] = False
    mock_game_state._set_points(0,0) # Start with 0 points

    player_turing_strategy = [[10],[20],[30]]
    
    with patch.object(manager, 'prepare_round_start_data', return_value="next_round_data") as mock_prepare:
        client_data, error = manager.submit_player_action(player_turing_strategy)

    assert error is None
    assert client_data == "next_round_data"
    
    # Check game.step was called correctly
    mock_game_state.step.assert_called_once_with(
        player_turing_strategy,
        manager.state["scherbius_planned_strategy"],
        manager.state["scherbius_planned_encryption"]
    )
    
    assert manager.state["last_round_summary"] is not None
    assert manager.state["last_round_summary"]["turing_points_gained_in_round"] > 0 # Based on mock step
    assert len(manager.state["round_history"]) == 1
    assert manager.state["round_history"][0]["round_number"] == 1
    
    mock_prepare.assert_called_once_with(is_new_round_for_player=not mock_game_state.is_won())

def test_submit_player_action_scherbius_player(manager, game_config, mock_game_state):
    manager.state["game_state"] = mock_game_state
    manager.state["player_role"] = "Scherbius"
    mock_game_state._set_points(0,0)
    # manager._turing_ai_fn is already mocked in the manager fixture to return [[2],[2]]

    player_scherbius_strategy = [[5],[5],[5]]
    player_scherbius_encrypts = True
    
    with patch.object(manager, 'prepare_round_start_data', return_value="next_round_data_s") as mock_prepare:
        client_data, error = manager.submit_player_action(player_scherbius_strategy, player_scherbius_encrypts)

    assert error is None
    assert client_data == "next_round_data_s"

    # AI Turing should have been called by determine_final_strategies
    manager._turing_ai_fn.assert_called_once_with(
        mock_game_state.turing_hand_cards, # Hand from turing_observation([])
        game_config.n_battles,
        game_config.max_cards_per_battle
    )
    
    # Check game.step was called correctly
    expected_turing_ai_strategy = manager._turing_ai_fn.return_value # [[2],[2]] from fixture
    mock_game_state.step.assert_called_once_with(
        expected_turing_ai_strategy,
        player_scherbius_strategy,
        player_scherbius_encrypts
    )
    
    assert manager.state["scherbius_planned_strategy"] == player_scherbius_strategy
    assert manager.state["scherbius_planned_encryption"] == player_scherbius_encrypts
    assert manager.state["turing_planned_strategy"] == expected_turing_ai_strategy
    assert manager.state["last_round_summary"] is not None
    # Based on mock step: S:[5,5,5] vs T:[2,2] (assuming n_battles=2 from mock AI)
    # If n_battles=3, T:[2,2,[]]. S wins 3 battles, gains 3 pts, loses 1 for encrypt. Net +2.
    # Let's adjust the mock AI to match n_battles
    manager._turing_ai_fn.return_value = [[2]] * game_config.n_battles # strat
    # Recalculate expected points: S plays [5,5,5], T plays [2,2,2]. S wins 3 battles (+3 pts). Encrypts (-1 pt). Total +2.
    # This depends on the mock_game_state.step logic.
    # Our mock_game_state.step: S gets 3 points, T gets 0. S encrypts, so S_final = 3 - 1 = 2.
    assert manager.state["last_round_summary"]["scherbius_points_gained_in_round"] == (game_config.n_battles * 1) - game_config.encryption_cost
    assert len(manager.state["round_history"]) == 1
    
    mock_prepare.assert_called_once_with(is_new_round_for_player=not mock_game_state.is_won())

def test_submit_player_action_turing_player_ai_scherbius_plan_missing(manager, mock_game_state):
    manager.state["game_state"] = mock_game_state
    manager.state["player_role"] = "Turing"
    manager.state["scherbius_planned_strategy"] = None # Simulate missing plan
    mock_game_state._set_won_state(False, "Null", 0, 0)

    player_turing_strategy = [[10],[20],[30]]
    client_data, error = manager.submit_player_action(player_turing_strategy)

    assert client_data is None
    assert error == "AI Scherbius plan missing. Please restart the game or report this bug."