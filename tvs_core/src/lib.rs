use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;

// Modules
pub mod enigma;
pub mod utils;
pub mod game_config;
pub mod game_types;
pub mod game_state;
pub mod game_logic;

// Re-export for easier access if needed, or keep them namespaced
use crate::game_config::PyGameConfig;
use crate::game_types::{Actor, Reward, Cards, ScherbiusAction, TuringAction, BattleOutcomeDetail};


#[pyclass(name = "GameState")] // Explicit Python name for PyGameState
struct PyGameState {
    inner: game_state::GameState, // Use the fully qualified type
    last_battle_outcomes: Vec<BattleOutcomeDetail>,
}

#[pymethods]
impl PyGameState {
    #[new]
    pub fn new(py_config: &PyGameConfig, seed: Option<u64>) -> PyResult<Self> {
        // PyGameConfig now holds an Arc<GameConfig> directly
        let game_config_arc = Arc::clone(&py_config.inner);
        let game_state = game_state::GameState::new(game_config_arc, seed);
        Ok(PyGameState {
            inner: game_state,
            last_battle_outcomes: Vec::new(),
        })
    }

    pub fn step(
        &mut self,
        turing_strategy: Vec<Cards>,
        scherbius_strategy: Vec<Cards>,
        reencrypt: bool,
    ) -> PyResult<()> {
        let scherbius_action = ScherbiusAction {
            strategy: scherbius_strategy,
            encryption: reencrypt,
        };
        let turing_action = TuringAction {
            strategy: turing_strategy,
        };

        // Clone the current state to pass by value to the logic function
        let current_inner_state = self.inner.clone();

        match game_logic::process_step(current_inner_state, &scherbius_action, &turing_action) {
            Ok((next_inner_state, outcomes)) => {
                self.inner = next_inner_state; // Update with the new state
                self.last_battle_outcomes = outcomes;
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }

    #[getter]
    pub fn battle_results(&self, py: Python) -> PyResult<Vec<Py<BattleOutcomeDetail>>> {
        let mut py_outcomes = Vec::new();
        for outcome_detail in &self.last_battle_outcomes {
            let py_outcome_detail = Py::new(py, outcome_detail.clone())?;
            py_outcomes.push(py_outcome_detail);
        }
        Ok(py_outcomes)
    }

    #[getter]
    pub fn is_won(&self) -> bool {
        self.inner.winner != Actor::Null
    }

    #[getter]
    pub fn turing_points(&self) -> u32 {
        self.inner.turing_points
    }

    #[getter]
    pub fn scherbius_points(&self) -> u32 {
        self.inner.scherbius_points
    }

    #[getter]
    pub fn turing_hand(&self) -> Vec<u32> {
        self.inner.turing_hand.clone()
    }

    #[getter]
    pub fn scherbius_hand(&self) -> Vec<u32> {
        self.inner.scherbius_hand.clone()
    }

    #[getter]
    pub fn winner(&self) -> &str {
        match self.inner.winner {
            Actor::Scherbius => "Scherbius",
            Actor::Turing => "Turing",
            Actor::Null => "Null",
        }
    }

    #[getter]
    pub fn rewards(&self) -> (Vec<Vec<u32>>, Vec<u32>) {
        let mut new_cards_rewards: Vec<Vec<u32>> = Vec::new();
        let mut victory_points_rewards: Vec<u32> = Vec::new();

        for reward in &self.inner.rewards { // Iterate over reference
            match reward {
                Reward::VictoryPoints(v) => {
                    victory_points_rewards.push(*v);
                    new_cards_rewards.push(vec![]);
                }
                Reward::NewCards(cards) => {
                    new_cards_rewards.push(cards.clone());
                    victory_points_rewards.push(0);
                }
                Reward::Null => {
                    new_cards_rewards.push(vec![]);
                    victory_points_rewards.push(0);
                }
            }
        }
        (new_cards_rewards, victory_points_rewards)
    }

    // This method mutates the encoder state within self.inner
    pub fn turing_observation(&mut self, scherbius_strategy: Vec<Cards>) -> (Vec<u32>, Vec<Cards>) {
        let intercepted_strategy = game_logic::intercept_scherbius_strategy(&mut self.inner, &scherbius_strategy);
        (self.inner.turing_hand.clone(), intercepted_strategy)
    }

    #[getter]
    pub fn scherbius_observation(&self) -> Vec<u32> {
        self.inner.scherbius_hand.clone()
    }

    // Add a way to get the raw GameState for testing or advanced use if needed (e.g., serialization)
    // This is optional and depends on your needs.
    // pub fn get_raw_state_json(&self) -> PyResult<String> {
    //     serde_json::to_string(&self.inner)
    //         .map_err(|e| PyValueError::new_err(format!("Failed to serialize state: {}", e)))
    // }
}

#[pymodule]
fn tvs_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<PyGameConfig>()?;
    m.add_class::<BattleOutcomeDetail>()?; // Add BattleOutcomeDetail to the module
    Ok(())
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*; // Access items from lib.rs
    use crate::game_config::GameConfig; // Explicitly use the Rust struct
    use crate::game_state::GameState;
    use crate::game_types::{ScherbiusAction, TuringAction, Reward, Actor, Action}; // Types
    use crate::game_logic; // For process_step and other logic functions
    use rand_chacha::ChaCha12Rng; // Add
    use rand::SeedableRng; // For StdRng::from_seed
    use std::sync::Arc;

    fn create_test_config(n_battles: u32, max_hand_size: u32, max_cards_per_battle: u32) -> Arc<GameConfig> {
        Arc::new(GameConfig {
            scherbius_starting: 5,
            turing_starting: 5,
            scherbius_deal: 2,
            turing_deal: 2,
            victory_points: 10,
            n_battles,
            encryption_cost: 3,
            encryption_vocab_size: 10,
            encryption_k_rotors: 2,
            max_vp: 3,
            max_draw: 3,
            verbose: false,
            max_hand_size,
            max_cards_per_battle,
        })
    }

    #[test]
    fn test_game_initialization() {
        let config = create_test_config(3, 10, 5);
        let game = GameState::new(Arc::clone(&config), None);

        assert_eq!(game.turing_points, 0);
        assert_eq!(game.scherbius_points, 0);
        assert_eq!(game.winner, Actor::Null);
        assert_eq!(game.turing_hand.len(), config.turing_starting as usize);
        assert_eq!(game.scherbius_hand.len(), config.scherbius_starting as usize);
        assert_eq!(game.rewards.len(), config.n_battles as usize);
        assert_eq!(game.game_config.max_hand_size, 10);
        assert_eq!(game.game_config.max_cards_per_battle, 5);
    }

    #[test]
    fn test_battle_result_logic() {
        assert_eq!(game_logic::battle_result(&vec![1, 2], &vec![3, 2]), Some(Actor::Turing));
        assert_eq!(game_logic::battle_result(&vec![5, 4], &vec![3, 2]), Some(Actor::Scherbius));
        assert_eq!(game_logic::battle_result(&vec![2, 3], &vec![3, 2]), None);
    }

    #[test]
    fn test_game_step_basic_flow_and_outcomes() {
        let n_battles = 2;
        let max_hand_size = 7;
        let max_cards_per_battle = 3;
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);
        
        let seed_val = 42; // Use a fixed seed for deterministic RNG in test
        // let mut initial_rng = StdRng::seed_from_u64(seed_val);
        let initial_rng = ChaCha12Rng::seed_from_u64(seed_val); // Use ChaCha12Rng

        // Initialize encoder with a clone of the RNG that will be put into GameState
        // to ensure its state is what we expect if it were initialized inside GameState::new
        // let _encoder_rng = initial_rng.clone(); // This variable is not used.

        let initial_game_state = GameState {
            turing_hand: vec![1, 2, 3, 4, 5, 6],
            scherbius_hand: vec![10, 20, 30, 40, 50, 60],
            turing_points: 0,
            scherbius_points: 0,
            encoder: enigma::EasyEnigma::new(
                config.encryption_vocab_size,
                config.encryption_k_rotors as usize,
                &mut initial_rng.clone(), // Pass a clone for encoder init
            ),
            winner: Actor::Null,
            rng: initial_rng, // This RNG will be used for rewards, card draws in step
            rewards: vec![Reward::VictoryPoints(2), Reward::NewCards(vec![100, 101])],
            game_config: Arc::clone(&config),
        };

        let scherbius_action = ScherbiusAction {
            strategy: vec![vec![10], vec![20, 30]],
            encryption: false,
        };
        let turing_action = TuringAction {
            strategy: vec![vec![1, 2], vec![3]],
        };

        match game_logic::process_step(initial_game_state, &scherbius_action, &turing_action) {
            Ok((next_game_state, outcomes)) => {
                assert_eq!(outcomes.len(), 2);
                // Battle 1: S(10) vs T(3). Scherbius wins. S gets 2 VP.
                assert_eq!(outcomes[0].scherbius_sum, 10);
                assert_eq!(outcomes[0].turing_sum, 3);
                assert_eq!(outcomes[0].turing_vp_won, 0);
                assert_eq!(outcomes[0].turing_cards_won.len(), 0);

                // Battle 2: S(50) vs T(3). Scherbius wins. S gets cards [100, 101].
                assert_eq!(outcomes[1].scherbius_sum, 50); // 20+30
                assert_eq!(outcomes[1].turing_sum, 3);
                assert_eq!(outcomes[1].turing_vp_won, 0);
                assert_eq!(outcomes[1].turing_cards_won.len(), 0);

                assert_eq!(next_game_state.scherbius_points, 2, "Scherbius points mismatch");
                assert_eq!(next_game_state.turing_points, 0, "Turing points mismatch");
                
                // Check hand sizes after drawing new cards (2 for each player by default config)
                // S played 3 cards (1+2), started with 6 -> 3. Won [100,101]. Dealt 2. Total: 3+2+2 = 7
                // T played 3 cards (2+1), started with 6 -> 3. Dealt 2. Total: 3+2 = 5
                assert_eq!(next_game_state.scherbius_hand.len(), 3 + 2 + config.scherbius_deal as usize);
                assert_eq!(next_game_state.turing_hand.len(), 3 + config.turing_deal as usize);

                assert!(next_game_state.scherbius_hand.len() <= max_hand_size as usize);
                assert!(next_game_state.turing_hand.len() <= max_hand_size as usize);
            }
            Err(e) => panic!("Step failed: {}", e),
        }
    }
    
    #[test]
    fn test_max_hand_size_truncation() {
        let n_battles = 1;
        let max_hand_size = 3; 
        let max_cards_per_battle = 1;
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);
        
        // Create a GameState where players will exceed max_hand_size after drawing
        let mut initial_game_state = GameState::new(Arc::clone(&config), Some(123)); 
        initial_game_state.scherbius_hand = vec![1,2,3,4,5]; // More than max_hand_size already
        initial_game_state.turing_hand = vec![1,2,3,4,5];
        // Ensure reward doesn't add too many cards, focus on deal + existing
        initial_game_state.rewards = vec![Reward::VictoryPoints(1)]; 

        let scherbius_action = ScherbiusAction { strategy: vec![vec![1]], encryption: false }; 
        let turing_action = TuringAction { strategy: vec![vec![2]] }; 

        match game_logic::process_step(initial_game_state, &scherbius_action, &turing_action) {
            Ok((next_state, _)) => {
                 // S played 1, had 5 -> 4. Dealt `scherbius_deal` (2). Total 6. Truncated to 3.
                 // T played 1, had 5 -> 4. Dealt `turing_deal` (2). Total 6. Truncated to 3.
                assert_eq!(next_state.scherbius_hand.len(), max_hand_size as usize, "Scherbius hand not truncated");
                assert_eq!(next_state.turing_hand.len(), max_hand_size as usize, "Turing hand not truncated");
            }
            Err(e) => panic!("Step failed: {}", e),
        }
    }

    #[test]
    fn test_max_cards_per_battle_validation() {
        let n_battles = 1;
        let max_hand_size = 5;
        let max_cards_per_battle = 2; 
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);

        let game_state = GameState::new(Arc::clone(&config), Some(124)); 
        // Ensure hands are sufficient for the test
        // game_state.scherbius_hand = vec![1,2,3,4,5]; // GameState::new already populates hands
        // game_state.turing_hand = vec![1,2,3,4,5];

        let valid_s_action = ScherbiusAction { strategy: vec![vec![game_state.scherbius_hand[0], game_state.scherbius_hand[1]]], encryption: false };
        let result = game_logic::check_action_validity(&game_state, &Action::ScherbiusAction(valid_s_action.clone()));
        assert!(result.is_ok());

        let invalid_s_cards = vec![game_state.scherbius_hand[0], game_state.scherbius_hand[1], game_state.scherbius_hand[2]];
        let invalid_s_action = ScherbiusAction { strategy: vec![invalid_s_cards], encryption: false }; 
        let result = game_logic::check_action_validity(&game_state, &Action::ScherbiusAction(invalid_s_action));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Too many cards"));
    }
    
    #[test]
    fn test_encryption_cost() {
        let config = create_test_config(1, 5, 2);
        let mut initial_state = GameState::new(Arc::clone(&config), Some(125)); 
        initial_state.scherbius_points = 5; 
        initial_state.scherbius_hand = vec![1,2,3]; // Ensure enough cards
        initial_state.turing_hand = vec![1,2,3];   
        initial_state.rewards = vec![Reward::VictoryPoints(1)]; // Turing wins battle, gets 1 VP

        let scherbius_action = ScherbiusAction { strategy: vec![vec![1]], encryption: true }; // S plays 1
        let turing_action = TuringAction { strategy: vec![vec![2]] }; // T plays 2, T wins

        let initial_s_points = initial_state.scherbius_points;
        let encryption_cost = initial_state.game_config.encryption_cost;

        match game_logic::process_step(initial_state, &scherbius_action, &turing_action) {
            Ok((next_state, _)) => {
                assert_eq!(next_state.scherbius_points, initial_s_points - encryption_cost);
                assert_eq!(next_state.turing_points, 1); 
            }
            Err(e) => panic!("Step failed: {}", e),
        }
    }

    #[test]
    fn test_strategy_must_cover_all_battles() {
        let n_battles = 2; 
        let config = create_test_config(n_battles, 5, 2);
        let game_state = GameState::new(Arc::clone(&config), Some(126)); 

        let scherbius_action_not_enough_battles = ScherbiusAction {
            strategy: vec![vec![game_state.scherbius_hand[0]]], // Only 1 battle strategy for 2 battles
            encryption: false,
        };
        let result = game_logic::check_action_validity(
            &game_state,
            &Action::ScherbiusAction(scherbius_action_not_enough_battles)
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Strategy must cover all"));
    }
    
    #[test]
    fn test_same_seed_produces_identical_gamestate() {
        let config = create_test_config(3, 10, 5); 
        let seed = Some(12345u64);

        let game1 = GameState::new(Arc::clone(&config), seed);
        let game2 = GameState::new(Arc::clone(&config), seed);

        assert_eq!(game1.turing_hand, game2.turing_hand, "Turing hands should be identical");
        assert_eq!(game1.scherbius_hand, game2.scherbius_hand, "Scherbius hands should be identical");
        assert_eq!(game1.rewards, game2.rewards, "Rewards lists should be identical");
        // Comparing raw wirings might be complex if they are Vec<Vec<u32>> etc.
        // Ensure get_rotor_wirings returns something comparable or compare relevant parts.
        assert_eq!(game1.encoder.get_rotor_wirings(), game2.encoder.get_rotor_wirings(), "Enigma rotors should be identical");
    }

    #[test]
    fn test_different_seeds_produce_different_gamestates() {
        let config = create_test_config(3, 10, 5);
        let seed1 = Some(101u64);
        let seed2 = Some(102u64); 

        assert_ne!(seed1, seed2, "Seeds must be different for this test");

        let game1 = GameState::new(Arc::clone(&config), seed1);
        let game2 = GameState::new(Arc::clone(&config), seed2);

        // These assertions are probabilistic but highly likely for different seeds
        assert_ne!(game1.turing_hand, game2.turing_hand, "Turing hands should differ (highly probable)");
        assert_ne!(game1.scherbius_hand, game2.scherbius_hand, "Scherbius hands should differ (highly probable)");
        
        // Rewards comparison
        let rewards_match = game1.rewards.len() == game2.rewards.len() && 
                            game1.rewards.iter().zip(game2.rewards.iter()).all(|(r1, r2)| r1 == r2);
        assert!(!rewards_match, "Rewards lists should differ (highly probable)");
        
        assert_ne!(game1.encoder.get_rotor_wirings(), game2.encoder.get_rotor_wirings(), "Enigma rotors should differ (highly probable)");
    }

    #[test]
    fn test_none_seed_initializes_gamestate() {
        let config = create_test_config(3, 10, 5);
        let game = GameState::new(Arc::clone(&config), None); 

        assert_eq!(game.turing_hand.len(), config.turing_starting as usize);
        assert_eq!(game.scherbius_hand.len(), config.scherbius_starting as usize);
        assert_eq!(game.rewards.len(), config.n_battles as usize);
        assert_eq!(game.turing_points, 0);
        assert_eq!(game.scherbius_points, 0);
        assert_eq!(game.winner, Actor::Null);
    }
}