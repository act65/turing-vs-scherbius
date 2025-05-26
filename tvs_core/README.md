The Python API for Turing vs Scherbius provides a bridge to the Rust implementation, allowing you to create and play the game from Python. Here's a comprehensive summary of the API:

## Core Classes

### PyGameConfig

This class configures the game parameters.

```Python
config = tvs.PyGameConfig(
    scherbius_starting=10,  # Number of cards Scherbius starts with
    scherbius_deal=2,       # Number of cards Scherbius gets each round
    turing_starting=10,     # Number of cards Turing starts with
    turing_deal=2,          # Number of cards Turing gets each round
    victory_points=10,      # Points needed to win
    n_battles=3,            # Number of battles per round
    encryption_cost=3,      # Cost in victory points to re-encrypt
    encryption_code_len=2,  # Length of encryption code
    encryption_vocab_size=10,  # Size of encryption vocabulary
    verbose=False,          # Verbose output flag
    max_vp=3,               # Maximum victory points in rewards
    max_draw=3              # Maximum number of cards in rewards
)
```

### PyGameState

This class represents the state of the game and provides methods to interact with it.

```python
game = tvs.PyGameState(config)  # Initialize with a PyGameConfig
```

## Game State Methods

### Basic State Accessors

```python
# Get current game state information
game.is_won()            # Returns True if game is over, False otherwise
game.winner()            # Returns "Turing", "Scherbius", or "Null" (game in progress)
game.turing_points()     # Returns Turing's current victory points
game.scherbius_points()  # Returns Scherbius's current victory points
game.encryption_broken() # Returns True if encryption is broken, False otherwise
```

### Player Hands
```python
# Get player hands
turing_hand = game.turing_hand()      # Returns list of Turing's cards
scherbius_hand = game.scherbius_hand() # Returns list of Scherbius's cards
```

### Game Rewards
```python

# Get current rewards
cards_rewards, vp_rewards = game.rewards()
# cards_rewards: List of lists of card rewards for each battle
# vp_rewards: List of victory point rewards for each battle
```

### Player Observations
```python
# Get Turing's observation (includes intercepted Scherbius strategy)
turing_hand, intercepted_strategy = game.turing_observation(scherbius_strategy)
# turing_hand: List of Turing's cards
# intercepted_strategy: List of intercepted Scherbius cards (encrypted if encryption not broken)

# Get Scherbius's observation
scherbius_hand = game.scherbius_observation()
# scherbius_hand: List of Scherbius's cards
```

### Game Actions


```python
# Execute a game step
game.step(
    turing_strategy,     # List of lists of cards Turing plays for each battle
    scherbius_strategy,  # List of lists of cards Scherbius plays for each battle
    turing_guesses,      # List of encryption code guesses by Turing
    reencrypt            # Boolean flag for Scherbius to re-encrypt
)
```

## Data Types

Cards: Represented as integers (usually 1-10)
Strategy: List of lists of cards, one list per battle (e.g., [[1, 2], [3, 4], []])
Encryption Guesses: List of potential encryption codes (e.g., [[1, 2], [3, 4]])
Rewards: Tuple of (card_rewards, victory_point_rewards)

### Example Usage

```python
# Create game
config = tvs.PyGameConfig(scherbius_starting=5, scherbius_deal=2, turing_starting=5,
                         turing_deal=2, victory_points=10, n_battles=3, encryption_cost=3,
                         encryption_code_len=2, encryption_vocab_size=10, verbose=False,
                         max_vp=3, max_draw=3)
game = tvs.PyGameState(config)

# Game loop
while not game.is_won():
    # Get current state
    turing_hand = game.turing_hand()
    scherbius_hand = game.scherbius_hand()
    
    # Player 1 (Scherbius) decides strategy
    scherbius_strategy = [[scherbius_hand[0], scherbius_hand[1]]]
    reencrypt = False
    
    # Player 2 (Turing) observes and decides strategy
    turing_hand, intercepted_scherbius = game.turing_observation(scherbius_strategy)
    turing_strategy = [[turing_hand[0], turing_hand[1]]]
    turing_guesses = [[turing_hand[2], turing_hand[3]]]
    
    # Execute game step
    game.step(turing_strategy, scherbius_strategy, turing_guesses, reencrypt)
    
    # Check rewards
    cards_rewards, vp_rewards = game.rewards()
    print(f"Rewards: {vp_rewards} VP, {cards_rewards} cards")
    
    # Print current score
    print(f"Score - Turing: {game.turing_points()}, Scherbius: {game.scherbius_points()}")
    print(f"Encryption broken: {game.encryption_broken()}")

# Print winner
print(f"Game over! Winner: {game.winner()}")
```

This API provides a flexible interface to implement various player strategies and game mechanics while leveraging the performance of the Rust implementation.