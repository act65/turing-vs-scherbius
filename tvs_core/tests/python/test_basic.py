import pytest
import turing_vs_scherbius as tvs

def test_game_creation():
    """Test that we can create a game with default configuration"""
    config = tvs.PyGameConfig(
        scherbius_starting=5,
        scherbius_deal=2,
        turing_starting=5,
        turing_deal=2,
        victory_points=10,
        n_battles=3,
        encryption_cost=3,
        encryption_code_len=2,
        encryption_vocab_size=10,
        verbose=False,
        max_vp=3,
        max_draw=3,
        max_cards_per_battle=3,
        max_hand_size=30
    )
    
    game = tvs.PyGameState(config, seed=42)
    
    # Check initial state
    assert game.is_won() == False
    assert game.turing_points() == 0
    assert game.scherbius_points() == 0
    assert len(game.turing_hand()) == 5
    assert len(game.scherbius_hand()) == 5
    assert game.winner() == "Null"

def test_game_step():
    """Test that a game step works as expected"""
    config = tvs.PyGameConfig(
        scherbius_starting=5,
        scherbius_deal=2,
        turing_starting=5,
        turing_deal=2,
        victory_points=10,
        n_battles=2,  # Expecting 2 battles
        encryption_cost=3,
        encryption_code_len=2,
        encryption_vocab_size=10,
        verbose=False,
        max_vp=3,
        max_draw=3,
        max_cards_per_battle=3, # Max 3 cards can be committed to a single battle
        max_hand_size=30
    )

    game = tvs.PyGameState(config)

    # Get initial hands
    turing_hand = game.turing_hand()
    scherbius_hand = game.scherbius_hand()

    # Ensure hands are not empty and have enough cards for the strategy
    assert len(turing_hand) >= 4, "Turing doesn't have enough cards for the strategy"
    assert len(scherbius_hand) >= 4, "Scherbius doesn't have enough cards for the strategy"

    # Create strategies for n_battles = 2
    # Example: Turing plays 2 cards in battle 1, 2 cards in battle 2
    # Scherbius plays 2 cards in battle 1, 2 cards in battle 2
    # Make sure these are distinct cards if that's the game rule (utils::is_subset_of_hand checks this)

    turing_strategy = [
        [turing_hand[0], turing_hand[1]],  # Cards for battle 1 (2 cards, <= max_cards_per_battle)
        [turing_hand[2], turing_hand[3]]   # Cards for battle 2 (2 cards, <= max_cards_per_battle)
    ]
    scherbius_strategy = [
        [scherbius_hand[0], scherbius_hand[1]], # Cards for battle 1
        [scherbius_hand[2], scherbius_hand[3]]  # Cards for battle 2
    ]

    try:
        game.step(turing_strategy, scherbius_strategy, False)
    except ValueError as e:
        pytest.fail(f"game.step raised ValueError unexpectedly: {e}")

    # Check that hands have changed (cards were played and new ones drawn)
    assert game.turing_hand() != turing_hand
    assert game.scherbius_hand() != scherbius_hand
    
    # Check that rewards exist
    cards_rewards, vp_rewards = game.rewards()
    assert len(cards_rewards) > 0
    assert len(vp_rewards) > 0

def test_observation():
    """Test that player observations work correctly"""
    config = tvs.PyGameConfig(
        scherbius_starting=5,
        scherbius_deal=2,
        turing_starting=5,
        turing_deal=2,
        victory_points=10,
        n_battles=3,
        encryption_cost=3,
        encryption_code_len=2,
        encryption_vocab_size=10,
        verbose=False,
        max_vp=3,
        max_draw=3,
        max_cards_per_battle=3,
        max_hand_size=30
    )
    
    game = tvs.PyGameState(config)
    
    # Get observations
    turing_hand, intercepted = game.turing_observation([[1, 2, 3]])
    scherbius_hand = game.scherbius_observation()
    
    # Check observation contents
    assert len(turing_hand) == 5
    assert isinstance(intercepted, list)
    assert len(intercepted) == 1
    assert isinstance(intercepted[0], list)
    assert len(scherbius_hand) == 5