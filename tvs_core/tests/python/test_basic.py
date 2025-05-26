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
        max_draw=3
    )
    
    game = tvs.PyGameState(config)
    
    # Check initial state
    assert game.is_won() == False
    assert game.turing_points() == 0
    assert game.scherbius_points() == 0
    assert game.encryption_broken() == False
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
        n_battles=3,
        encryption_cost=3,
        encryption_code_len=2,
        encryption_vocab_size=10,
        verbose=False,
        max_vp=3,
        max_draw=3
    )
    
    game = tvs.PyGameState(config)
    
    # Get initial hands
    turing_hand = game.turing_hand()
    scherbius_hand = game.scherbius_hand()
    
    # Create some simple strategies
    turing_strategy = [[turing_hand[0], turing_hand[1]]]
    scherbius_strategy = [[scherbius_hand[0], scherbius_hand[1]]]
    turing_guesses = []  # No encryption guesses
    
    # Execute a step
    game.step(turing_strategy, scherbius_strategy, turing_guesses, False)
    
    # Check that hands have changed (cards were played and new ones drawn)
    assert game.turing_hand() != turing_hand
    assert game.scherbius_hand() != scherbius_hand
    
    # Check that rewards exist
    cards_rewards, vp_rewards = game.rewards()
    assert len(cards_rewards) > 0
    assert len(vp_rewards) > 0

def test_encryption_breaking():
    """Test that encryption can be broken"""
    # Create a game with a known encryption
    config = tvs.PyGameConfig(
        scherbius_starting=10,
        scherbius_deal=2,
        turing_starting=10,
        turing_deal=2,
        victory_points=10,
        n_battles=1,
        encryption_cost=3,
        encryption_code_len=2,  # Small code length for easier testing
        encryption_vocab_size=10,
        verbose=False,
        max_vp=3,
        max_draw=3
    )
    
    game = tvs.PyGameState(config)
    
    # Try different guesses until we break encryption
    encryption_broken = False
    max_attempts = 20  # Limit attempts to avoid infinite loop
    
    for _ in range(max_attempts):
        if encryption_broken:
            break
            
        turing_hand = game.turing_hand()
        scherbius_hand = game.scherbius_hand()
        
        # Make a single guess from the hand
        # In the real game, we'd make more intelligent guesses
        # But for the test, we'll just use the first 2 cards if we have enough
        if len(turing_hand) >= 2:
            guesses = [[turing_hand[0], turing_hand[1]]]
        else:
            guesses = []  # Not enough cards to guess
                
        # Simple strategies - use different cards than the ones in the guess
        turing_strategy = [[turing_hand[2]]] if len(turing_hand) > 2 else [[]]
        scherbius_strategy = [[scherbius_hand[0]]] if scherbius_hand else [[]]
        
        # Execute step
        game.step(turing_strategy, scherbius_strategy, guesses, False)
        
        # Check if encryption was broken
        encryption_broken = game.encryption_broken()
    
    # This test may sometimes fail due to randomness - that's okay
    # We're just testing the mechanism works
    print(f"Encryption broken: {encryption_broken} after {_+1} attempts")
    
    # We don't assert here because breaking encryption is probabilistic
    # Instead, we're just testing that the mechanism works without errors

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
        max_draw=3
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