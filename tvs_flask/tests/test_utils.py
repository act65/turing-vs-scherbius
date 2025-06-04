import pytest
import random
from unittest.mock import patch, MagicMock
import utils # Assuming this is in PYTHONPATH or same directory
from .conftest import MockGameConfig, MockPyGameState, MockBattleOutcome # Import mocks

@pytest.fixture
def config():
    return MockGameConfig()

def test_get_initial_state():
    state = utils.get_initial_state()
    assert isinstance(state, dict)
    expected_keys = [
        "game_state", "player_role", "scherbius_planned_strategy",
        "scherbius_planned_encryption", "turing_planned_strategy",
        "last_round_summary", "round_history", "current_round_potential_rewards",
        "last_client_data_prepared", "player_initial_hand_for_turn"
    ]
    for key in expected_keys:
        assert key in state
    assert state["round_history"] == []
    assert state["player_initial_hand_for_turn"] == []

@patch('random.shuffle', side_effect=lambda x: x) # No-op shuffle
@patch('random.random')
@patch('random.randint')
@patch('random.choice')
def test_scherbius_ai_player_logic(mock_choice, mock_randint, mock_random, mock_shuffle, config):
    mock_random.return_value = 0.8 # Ensure cards are played (threshold 0.9)
    mock_randint.return_value = 1  # Play 1 card per battle if possible
    mock_choice.return_value = True # Encrypt

    hand = [10, 20, 30, 40]
    strategy, encrypt = utils.scherbius_ai_player_logic(hand, config.n_battles, config.max_cards_per_battle)

    assert len(strategy) == config.n_battles
    assert encrypt is True
    
    total_cards_played = sum(len(b) for b in strategy)
    # Expects to play 1 card for each of n_battles, or until hand runs out
    assert total_cards_played <= len(hand)
    assert total_cards_played <= config.n_battles * config.max_cards_per_battle

    # Check cards are from hand (mock_shuffle is no-op, so order is reversed due to pop)
    # This part is tricky due to random.pop() and shuffle.
    # A simpler check is that played cards were in the original hand.
    played_cards_flat = [card for BATTLE in strategy for card in BATTLE]
    for card in played_cards_flat:
        assert card in hand
    
    # Test with empty hand
    strategy_empty, _ = utils.scherbius_ai_player_logic([], config.n_battles, config.max_cards_per_battle)
    assert sum(len(b) for b in strategy_empty) == 0

@patch('random.shuffle', side_effect=lambda x: x)
@patch('random.random')
@patch('random.randint')
def test_turing_ai_player_logic(mock_randint, mock_random, mock_shuffle, config):
    mock_random.return_value = 0.6 # Ensure cards are played (threshold 0.7)
    mock_randint.return_value = 1

    hand = [5, 15, 25]
    strategy = utils.turing_ai_player_logic(hand, config.n_battles, config.max_cards_per_battle)

    assert len(strategy) == config.n_battles
    total_cards_played = sum(len(b) for b in strategy)
    assert total_cards_played <= len(hand)
    assert total_cards_played <= config.n_battles * config.max_cards_per_battle
    
    played_cards_flat = [card for BATTLE in strategy for card in BATTLE]
    for card in played_cards_flat:
        assert card in hand

def test_handle_pre_turn_observations_and_ai(config):
    mock_game = MockPyGameState(config)
    mock_scherbius_ai_fn = MagicMock(return_value=([[1],[2],[3]], True)) # strategy, encrypt

    # Test for Turing player
    result_turing = utils.handle_pre_turn_observations_and_ai(
        mock_game, "Turing", config, mock_scherbius_ai_fn
    )
    assert result_turing["current_phase"] == "Turing_Action"
    mock_scherbius_ai_fn.assert_called_once_with(mock_game.scherbius_hand_cards, config.n_battles, config.max_cards_per_battle)
    assert result_turing["scherbius_planned_strategy"] == [[1],[2],[3]]
    assert result_turing["scherbius_planned_encryption"] is True
    assert result_turing["scherbius_encryption_status_for_view"] is True
    assert result_turing["current_player_hand_values"] == mock_game.turing_hand_cards
    # Check intercepted plays based on mock_game.turing_observation logic
    expected_intercepted = [[] for _ in range(config.n_battles)]
    if [[1],[2],[3]] and [[1],[2],[3]][0]: # from mock_game.turing_observation
        expected_intercepted[0] = [[[1],[2],[3]][0][0]] # first card of first battle
    assert result_turing["opponent_observed_plays_for_player"] == expected_intercepted


    # Test for Scherbius player
    mock_scherbius_ai_fn.reset_mock()
    result_scherbius = utils.handle_pre_turn_observations_and_ai(
        mock_game, "Scherbius", config, mock_scherbius_ai_fn
    )
    assert result_scherbius["current_phase"] == "Scherbius_Action"
    mock_scherbius_ai_fn.assert_not_called() # AI Scherbius doesn't plan if player is Scherbius here
    assert result_scherbius["scherbius_planned_strategy"] is None
    assert result_scherbius["scherbius_planned_encryption"] is False
    assert result_scherbius["scherbius_encryption_status_for_view"] is False
    assert result_scherbius["current_player_hand_values"] == mock_game.scherbius_hand_cards
    assert result_scherbius["opponent_observed_plays_for_player"] == [[] for _ in range(config.n_battles)]

def test_prepare_player_hand_for_display():
    current_hand_values = [10, 20]
    # New round
    hand_client, hand_state = utils.prepare_player_hand_for_display(current_hand_values, [], True)
    assert len(hand_client) == 2
    assert hand_client[0] == {"id": "pcard_0", "value": 10}
    assert hand_client[1] == {"id": "pcard_1", "value": 20}
    assert hand_client == hand_state

    # Existing round, existing hand
    existing_initial_hand = [{"id": "old_0", "value": 5}, {"id": "old_1", "value": 15}]
    hand_client, hand_state = utils.prepare_player_hand_for_display(
        current_hand_values, existing_initial_hand, False
    )
    assert hand_client == existing_initial_hand # Should return existing
    assert hand_state == existing_initial_hand

    # Existing round, but no existing initial hand (e.g. first call in existing round)
    hand_client, hand_state = utils.prepare_player_hand_for_display(current_hand_values, [], False)
    assert len(hand_client) == 2
    assert hand_client[0] == {"id": "pcard_0", "value": 10}
    assert hand_client == hand_state


def test_assemble_client_data_for_round_start(config):
    mock_game = MockPyGameState(config)
    mock_game._set_points(5,3)
    
    game_state_snapshot = {"last_round_summary": {"details": "some"}, "round_history": ["hist1"]}
    player_hand_client = [{"id": "pcard_0", "value": 10}]
    opponent_observed = [[1]]
    scherbius_encrypt_view = True
    current_phase = "Turing_Action"

    client_data = utils.assemble_client_data_for_round_start(
        mock_game, "Turing", config, game_state_snapshot,
        player_hand_client, opponent_observed, scherbius_encrypt_view, current_phase
    )

    assert client_data["player_role"] == "Turing"
    assert client_data["player_hand"] == player_hand_client
    assert client_data["opponent_observed_plays"] == opponent_observed
    assert client_data["scherbius_did_encrypt"] == scherbius_encrypt_view
    assert client_data["rewards"]["card_rewards"] == mock_game.current_card_rewards
    assert client_data["rewards"]["vp_rewards"] == mock_game.current_vp_rewards
    assert client_data["turing_points"] == 5
    assert client_data["scherbius_points"] == 3
    assert client_data["is_game_over"] == mock_game.is_won()
    assert client_data["last_round_summary"] == game_state_snapshot["last_round_summary"]
    assert client_data["current_phase"] == current_phase
    assert client_data["round_history"] == game_state_snapshot["round_history"]
    # Check a few config values
    assert client_data["n_battles"] == config.n_battles
    assert client_data["max_victory_points"] == config.victory_points


def test_determine_final_strategies(config):
    mock_game = MockPyGameState(config)
    mock_turing_ai_fn = MagicMock(return_value=([[10],[20],[30]])) # AI Turing's strategy

    player_strat = [[1],[2],[3]]
    
    # Player is Turing
    s_planned_strat = [[5],[6],[7]]
    s_planned_encrypt = True
    result_turing = utils.determine_final_strategies(
        mock_game, "Turing", config, player_strat, None,
        s_planned_strat, s_planned_encrypt, mock_turing_ai_fn
    )
    assert result_turing["final_turing_strategy"] == player_strat
    assert result_turing["final_scherbius_strategy"] == s_planned_strat
    assert result_turing["scherbius_encryption_for_step"] == s_planned_encrypt
    assert result_turing["updated_scherbius_planned_strategy"] == s_planned_strat # Unchanged
    assert result_turing["updated_scherbius_planned_encryption"] == s_planned_encrypt # Unchanged
    assert result_turing["turing_planned_strategy_for_state"] is None
    mock_turing_ai_fn.assert_not_called()

    # Player is Scherbius
    mock_turing_ai_fn.reset_mock()
    s_encrypts_player = True
    result_scherbius = utils.determine_final_strategies(
        mock_game, "Scherbius", config, player_strat, s_encrypts_player,
        None, False, mock_turing_ai_fn # Initial Scherbius AI plan not relevant here
    )
    assert result_scherbius["final_scherbius_strategy"] == player_strat
    assert result_scherbius["scherbius_encryption_for_step"] == s_encrypts_player
    mock_turing_ai_fn.assert_called_once_with(mock_game.turing_hand_cards, config.n_battles, config.max_cards_per_battle)
    assert result_scherbius["final_turing_strategy"] == [[10],[20],[30]] # From mock AI
    assert result_scherbius["updated_scherbius_planned_strategy"] == player_strat
    assert result_scherbius["updated_scherbius_planned_encryption"] == s_encrypts_player
    assert result_scherbius["turing_planned_strategy_for_state"] == [[10],[20],[30]]


def test_create_last_round_summary_data(config):
    summary = utils.create_last_round_summary_data(
        5, 2, 8, 3, # prev_t, prev_s, curr_t, curr_s
        [[1],[2]], [[10]], True, # t_strat, s_strat, encrypted, n_battles (use 2 for this test)
        2 # n_battles = 2
    )
    assert summary["turing_points_gained_in_round"] == 3
    assert summary["scherbius_points_gained_in_round"] == 1
    assert summary["scherbius_encrypted_last_round"] is True
    assert len(summary["battle_details"]) == 2
    assert summary["battle_details"][0]["turing_played"] == [1]
    assert summary["battle_details"][0]["scherbius_committed"] == [10]

def test_create_historical_round_entry_data(config):
    battle_outcomes = [
        MockBattleOutcome(turing_sum=10, scherbius_sum=5),
        MockBattleOutcome(turing_sum=3, scherbius_sum=8),
        MockBattleOutcome(turing_sum=7, scherbius_sum=7)
    ]
    potential_rewards = {
        "card_rewards": [[11], [22], [33]],
        "vp_rewards": [1, 2, 3]
    }
    entry = utils.create_historical_round_entry_data(
        1, 8, 3, True, battle_outcomes, # round_num, t_pts, s_pts, encrypted
        [[10],[3],[7]], [[5],[8],[7]], # t_strat, s_strat
        potential_rewards, config.n_battles
    )
    assert entry["round_number"] == 1
    assert entry["turing_total_points_after_round"] == 8
    assert entry["scherbius_total_points_after_round"] == 3
    assert entry["scherbius_encrypted_this_round"] is True
    assert len(entry["battles"]) == config.n_battles
    assert entry["battles"][0]["winner"] == "Turing"
    assert entry["battles"][1]["winner"] == "Scherbius"
    assert entry["battles"][2]["winner"] == "Draw"
    assert entry["battles"][0]["turing_played_cards"] == [10]
    assert entry["battles"][0]["scherbius_committed_cards"] == [5]
    assert entry["battles"][0]["rewards_available_to_turing"]["vp"] == potential_rewards["vp_rewards"][0]
    assert entry["battles"][0]["rewards_available_to_turing"]["cards"] == potential_rewards["card_rewards"][0]