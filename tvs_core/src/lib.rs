// src/lib.rs

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
// game_state::GameState is used internally by PyGameState

#[pyclass(name = "GameState")] // Explicit Python name for PyGameState
struct PyGameState {
    inner: game_state::GameState, // Use the fully qualified type
    // last_battle_outcomes: Vec<BattleOutcomeDetail>, // REMOVED
}

#[pymethods]
impl PyGameState {
    #[new]
    pub fn new(py_config: &PyGameConfig, seed: Option<u64>) -> PyResult<Self> {
        let game_config_arc = Arc::clone(&py_config.inner);
        let game_state = game_state::GameState::new(game_config_arc, seed);
        Ok(PyGameState {
            inner: game_state,
            // last_battle_outcomes: Vec::new(), // REMOVED
        })
    }

    // step method REMOVED
    // battle_results getter REMOVED

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
    pub fn winner(&self) -> &str { // Returns &str, which is fine for PyO3
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

        for reward in &self.inner.rewards {
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

    // turing_observation method REMOVED (replaced by py_intercept_scherbius_strategy)

    #[getter]
    pub fn scherbius_observation(&self) -> Vec<u32> {
        self.inner.scherbius_hand.clone()
    }
}

#[pyfunction]
fn py_process_step(
    py: Python,
    current_py_state: &PyGameState, // Pass by reference, clone inner state
    turing_strategy: Vec<Cards>,
    scherbius_strategy: Vec<Cards>,
    reencrypt: bool,
) -> PyResult<(Py<PyGameState>, Vec<Py<BattleOutcomeDetail>>)> {
    let scherbius_action = ScherbiusAction {
        strategy: scherbius_strategy,
        encryption: reencrypt,
    };
    let turing_action = TuringAction {
        strategy: turing_strategy,
    };

    let current_inner_state = current_py_state.inner.clone();

    match game_logic::process_step(current_inner_state, &scherbius_action, &turing_action) {
        Ok((next_inner_state, outcomes_rust)) => {
            let next_py_state_obj = PyGameState {
                inner: next_inner_state,
            };
            let py_next_state = Py::new(py, next_py_state_obj)?;

            let mut py_outcomes = Vec::with_capacity(outcomes_rust.len());
            for outcome_detail_rust in outcomes_rust {
                let py_outcome_detail = Py::new(py, outcome_detail_rust.clone())?;
                py_outcomes.push(py_outcome_detail);
            }
            Ok((py_next_state, py_outcomes))
        }
        Err(e) => Err(PyValueError::new_err(e)), // e is already String from game_logic
    }
}

#[pyfunction]
fn py_intercept_scherbius_strategy(
    py: Python,
    state: &PyGameState, // Now an immutable reference
    scherbius_strategy: Vec<Cards>,
) -> PyResult<(Py<PyGameState>, Vec<Cards>)> {
    // Call the modified game_logic function
    let (new_encoder, intercepted_cards) =
        game_logic::intercept_scherbius_strategy(&state.inner, &scherbius_strategy);

    // Create a new PyGameState
    let mut new_inner_game_state = state.inner.clone();
    new_inner_game_state.encoder = new_encoder;

    let new_py_game_state = Py::new(py, PyGameState { inner: new_inner_game_state })?;

    Ok((new_py_game_state, intercepted_cards))
}


#[pymodule]
fn tvs_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<PyGameConfig>()?;
    m.add_class::<BattleOutcomeDetail>()?;

    m.add_function(wrap_pyfunction!(py_process_step, m)?)?;
    m.add_function(wrap_pyfunction!(py_intercept_scherbius_strategy, m)?)?;
    Ok(())
}

// --- Tests ---
// The Rust tests should largely remain the same as they test the core Rust logic.
// Ensure GameState and EasyEnigma derive Clone.
// (Your existing tests seem to correctly use game_logic::process_step directly)
#[cfg(test)]
mod tests {
    use super::*; // Access items from lib.rs
    use crate::game_config::GameConfig; // Explicitly use the Rust struct
    use crate::game_state::GameState; // The Rust struct
    use crate::game_types::{ScherbiusAction, TuringAction, Reward, Actor, Action}; // Types
    use crate::game_logic; // For process_step and other logic functions
    use rand_chacha::ChaCha12Rng;
    use rand::SeedableRng;
    use std::sync::Arc;
    use crate::enigma::EasyEnigma; // Ensure EasyEnigma is in scope for tests

    // create_test_config remains the same
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

    // test_game_initialization remains the same
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

    // test_battle_result_logic remains the same
    #[test]
    fn test_battle_result_logic() {
        assert_eq!(game_logic::battle_result(&vec![1, 2], &vec![3, 2]), Some(Actor::Turing)); // T sum 3, S sum 5 -> S wins
        // Corrected logic: battle_result returns winner of the *battle*, not overall game.
        // battle_result(&s_cards, &t_cards)
        // S:[1,2] (sum 3), T:[3,2] (sum 5) -> Turing wins
        assert_eq!(game_logic::battle_result(&vec![1, 2], &vec![3, 2]), Some(Actor::Turing));
        // S:[5,4] (sum 9), T:[3,2] (sum 5) -> Scherbius wins
        assert_eq!(game_logic::battle_result(&vec![5, 4], &vec![3, 2]), Some(Actor::Scherbius));
        // S:[2,3] (sum 5), T:[3,2] (sum 5) -> Draw (None)
        assert_eq!(game_logic::battle_result(&vec![2, 3], &vec![3, 2]), None);
    }
    
    // test_game_step_basic_flow_and_outcomes:
    // This test uses the Rust `GameState` and `game_logic::process_step` directly.
    // The assertions on `outcomes[X].field` will work because `BattleOutcomeDetail` (Rust struct)
    // has public fields with these names.
    // The `initial_game_state.encoder` assignment is fine if `EasyEnigma` is part of `GameState` and public.
    // This test should still pass.
    #[test]
    fn test_game_step_basic_flow_and_outcomes() {
        let n_battles = 2;
        let max_hand_size = 7;
        let max_cards_per_battle = 3;
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);
        
        let seed_val = 42;
        let initial_rng = ChaCha12Rng::seed_from_u64(seed_val);

        let initial_game_state = GameState {
            turing_hand: vec![1, 2, 3, 4, 5, 6], // T plays 1,2 then 3. Sums: 3, 3
            scherbius_hand: vec![10, 20, 30, 40, 50, 60], // S plays 10, then 20,30. Sums: 10, 50
            turing_points: 0,
            scherbius_points: 0,
            encoder: enigma::EasyEnigma::new( // Ensure EasyEnigma::new is available and correct
                config.encryption_vocab_size,
                config.encryption_k_rotors as usize,
                &mut initial_rng.clone(), 
            ),
            winner: Actor::Null,
            rng: initial_rng, 
            rewards: vec![Reward::VictoryPoints(2), Reward::NewCards(vec![100, 101])], // S wins battle 1 (gets 2 VP), S wins battle 2 (gets cards)
            game_config: Arc::clone(&config),
        };

        let scherbius_action = ScherbiusAction {
            strategy: vec![vec![10], vec![20, 30]], // Sums: 10, 50
            encryption: false,
        };
        let turing_action = TuringAction {
            strategy: vec![vec![1, 2], vec![3]], // Sums: 3, 3
        };

        match game_logic::process_step(initial_game_state.clone(), &scherbius_action, &turing_action) {
            Ok((next_game_state, outcomes)) => {
                assert_eq!(outcomes.len(), 2);
                // Battle 1: S(10) vs T(3). Scherbius wins. S gets Reward[0] = 2 VP.
                assert_eq!(outcomes[0].scherbius_sum, 10);
                assert_eq!(outcomes[0].turing_sum, 3);
                assert_eq!(outcomes[0].battle_winner, Actor::Scherbius);
                assert_eq!(outcomes[0].scherbius_vp_won, 2); // From Reward::VictoryPoints(2)
                assert_eq!(outcomes[0].scherbius_cards_won.len(), 0);
                assert_eq!(outcomes[0].turing_vp_won, 0);
                assert_eq!(outcomes[0].turing_cards_won.len(), 0);

                // Battle 2: S(50) vs T(3). Scherbius wins. S gets Reward[1] = cards [100, 101].
                assert_eq!(outcomes[1].scherbius_sum, 50);
                assert_eq!(outcomes[1].turing_sum, 3);
                assert_eq!(outcomes[1].battle_winner, Actor::Scherbius);
                assert_eq!(outcomes[1].scherbius_vp_won, 0);
                assert_eq!(outcomes[1].scherbius_cards_won, vec![100, 101]); // From Reward::NewCards
                assert_eq!(outcomes[1].turing_vp_won, 0);
                assert_eq!(outcomes[1].turing_cards_won.len(), 0);
                
                assert_eq!(next_game_state.scherbius_points, 2, "Scherbius points mismatch");
                assert_eq!(next_game_state.turing_points, 0, "Turing points mismatch");
                
                // Hands after step:
                // Initial S hand: 6 cards. Played: 1 (for battle 1) + 2 (for battle 2) = 3 cards. Remaining: 6 - 3 = 3.
                // S won cards [100, 101]. Hand becomes 3 + 2 = 5.
                // S dealt `config.scherbius_deal` (2) cards. Hand becomes 5 + 2 = 7.
                // Max hand size is 7. So, 7.
                assert_eq!(next_game_state.scherbius_hand.len(), 7.min(max_hand_size as usize));
                
                // Initial T hand: 6 cards. Played: 2 (for battle 1) + 1 (for battle 2) = 3 cards. Remaining: 6 - 3 = 3.
                // T won no cards.
                // T dealt `config.turing_deal` (2) cards. Hand becomes 3 + 2 = 5.
                // Max hand size is 7. So, 5.
                assert_eq!(next_game_state.turing_hand.len(), 5.min(max_hand_size as usize));

                assert!(next_game_state.scherbius_hand.len() <= max_hand_size as usize);
                assert!(next_game_state.turing_hand.len() <= max_hand_size as usize);
            }
            Err(e) => panic!("Step failed: {}", e),
        }
    }

    // test_max_hand_size_truncation remains the same
    #[test]
    fn test_max_hand_size_truncation() {
        let n_battles = 1;
        let max_hand_size = 3; 
        let max_cards_per_battle = 1;
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);
        
        let mut initial_game_state = GameState::new(Arc::clone(&config), Some(123)); 
        initial_game_state.scherbius_hand = vec![1,2,3,4,5]; 
        initial_game_state.turing_hand = vec![1,2,3,4,5];
        initial_game_state.rewards = vec![Reward::VictoryPoints(1)]; 

        let scherbius_action = ScherbiusAction { strategy: vec![vec![initial_game_state.scherbius_hand[0]]], encryption: false }; 
        let turing_action = TuringAction { strategy: vec![vec![initial_game_state.turing_hand[0]]] }; 

        match game_logic::process_step(initial_game_state, &scherbius_action, &turing_action) {
            Ok((next_state, _)) => {
                assert_eq!(next_state.scherbius_hand.len(), max_hand_size as usize, "Scherbius hand not truncated");
                assert_eq!(next_state.turing_hand.len(), max_hand_size as usize, "Turing hand not truncated");
            }
            Err(e) => panic!("Step failed: {}", e),
        }
    }

    // test_max_cards_per_battle_validation remains the same
    #[test]
    fn test_max_cards_per_battle_validation() {
        let n_battles = 1;
        let max_hand_size = 5;
        let max_cards_per_battle = 2; 
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);

        let mut game_state = GameState::new(Arc::clone(&config), Some(124)); 
        // Ensure hands are sufficient for the test by re-assigning if necessary, GameState::new populates them based on config.scherbius_starting
        game_state.scherbius_hand = vec![1,2,3,4,5]; 
        game_state.turing_hand = vec![1,2,3,4,5];


        let valid_s_action = ScherbiusAction { strategy: vec![vec![game_state.scherbius_hand[0], game_state.scherbius_hand[1]]], encryption: false };
        let result = game_logic::check_action_validity(&game_state, &Action::ScherbiusAction(valid_s_action.clone()));
        assert!(result.is_ok());

        let invalid_s_cards = vec![game_state.scherbius_hand[0], game_state.scherbius_hand[1], game_state.scherbius_hand[2]];
        let invalid_s_action = ScherbiusAction { strategy: vec![invalid_s_cards], encryption: false }; 
        let result = game_logic::check_action_validity(&game_state, &Action::ScherbiusAction(invalid_s_action));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Too many cards"));
    }
    
    // test_encryption_cost remains the same
    #[test]
    fn test_encryption_cost() {
        let config = create_test_config(1, 5, 2); // encryption_cost = 3 by default in create_test_config
        let mut initial_state = GameState::new(Arc::clone(&config), Some(125)); 
        initial_state.scherbius_points = 5; 
        initial_state.scherbius_hand = vec![10,20,30]; 
        initial_state.turing_hand = vec![1,2,3];   
        initial_state.rewards = vec![Reward::VictoryPoints(1)]; // Turing wins battle (S:10 vs T:1), T gets 1 VP

        let scherbius_action = ScherbiusAction { strategy: vec![vec![10]], encryption: true }; 
        let turing_action = TuringAction { strategy: vec![vec![1]] }; // T plays 1, S plays 10. S wins battle.

        // Re-evaluate scenario for encryption cost:
        // S starts with 5 points. Encrypts (cost 3). S points -> 2.
        // Battle: S(10) vs T(1). S wins. Reward is VP(1) for S. S points -> 2 + 1 = 3.
        // T points -> 0.
        initial_state.rewards = vec![Reward::VictoryPoints(1)]; // Winner of battle gets 1 VP.

        let initial_s_points = initial_state.scherbius_points; // 5
        let encryption_cost = initial_state.game_config.encryption_cost; // 3

        match game_logic::process_step(initial_state.clone(), &scherbius_action, &turing_action) {
            Ok((next_state, outcomes)) => {
                assert_eq!(outcomes[0].battle_winner, Actor::Scherbius);
                assert_eq!(outcomes[0].scherbius_vp_won, 1);
                assert_eq!(next_state.scherbius_points, initial_s_points - encryption_cost + 1); // 5 - 3 + 1 = 3
                assert_eq!(next_state.turing_points, 0); 
            }
            Err(e) => panic!("Step failed: {}", e),
        }
    }

    // test_strategy_must_cover_all_battles remains the same
    #[test]
    fn test_strategy_must_cover_all_battles() {
        let n_battles = 2; 
        let config = create_test_config(n_battles, 5, 2);
        let mut game_state = GameState::new(Arc::clone(&config), Some(126)); 
        game_state.scherbius_hand = vec![1,2,3,4,5]; // Ensure hand has cards

        let scherbius_action_not_enough_battles = ScherbiusAction {
            strategy: vec![vec![game_state.scherbius_hand[0]]], 
            encryption: false,
        };
        let result = game_logic::check_action_validity(
            &game_state,
            &Action::ScherbiusAction(scherbius_action_not_enough_battles)
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Strategy must cover all"));
    }
    
    // test_same_seed_produces_identical_gamestate remains the same
    #[test]
    fn test_same_seed_produces_identical_gamestate() {
        let config = create_test_config(3, 10, 5); 
        let seed = Some(12345u64);

        let game1 = GameState::new(Arc::clone(&config), seed);
        let game2 = GameState::new(Arc::clone(&config), seed);

        assert_eq!(game1.turing_hand, game2.turing_hand, "Turing hands should be identical");
        assert_eq!(game1.scherbius_hand, game2.scherbius_hand, "Scherbius hands should be identical");
        assert_eq!(game1.rewards, game2.rewards, "Rewards lists should be identical");
        assert_eq!(game1.encoder.get_rotor_wirings(), game2.encoder.get_rotor_wirings(), "Enigma rotors should be identical");
    }

    // test_different_seeds_produce_different_gamestates remains the same
    #[test]
    fn test_different_seeds_produce_different_gamestates() {
        let config = create_test_config(3, 10, 5);
        let seed1 = Some(101u64);
        let seed2 = Some(102u64); 

        assert_ne!(seed1, seed2, "Seeds must be different for this test");

        let game1 = GameState::new(Arc::clone(&config), seed1);
        let game2 = GameState::new(Arc::clone(&config), seed2);
        
        assert_ne!(game1.turing_hand, game2.turing_hand, "Turing hands should differ (highly probable)");
        assert_ne!(game1.scherbius_hand, game2.scherbius_hand, "Scherbius hands should differ (highly probable)");
        
        let rewards_match = game1.rewards.len() == game2.rewards.len() && 
                            game1.rewards.iter().zip(game2.rewards.iter()).all(|(r1, r2)| r1 == r2);
        assert!(!rewards_match, "Rewards lists should differ (highly probable)");
        
        assert_ne!(game1.encoder.get_rotor_wirings(), game2.encoder.get_rotor_wirings(), "Enigma rotors should differ (highly probable)");
    }

    // test_none_seed_initializes_gamestate remains the same
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

    #[test]
    fn test_intercept_scherbius_strategy_clones_and_returns_encoder() {
        let config = create_test_config(1, 5, 2); // n_battles, max_hand_size, max_cards_per_battle
        let initial_game_state = GameState::new(Arc::clone(&config), Some(999));

        // Clone the original encoder for later comparison
        let original_encoder_clone = initial_game_state.encoder.clone();
        let original_steps_before_call = original_encoder_clone.get_steps().clone(); // Clone to avoid borrowing issues
        let original_wirings_before_call = original_encoder_clone.get_rotor_wirings().clone();

        let scherbius_strategy: Vec<Cards> = vec![vec![1, 2], vec![3]]; // Example strategy

        // Execute the function
        let (returned_encoder, intercepted_strategy) =
            game_logic::intercept_scherbius_strategy(&initial_game_state, &scherbius_strategy);

        // Assertions
        assert_eq!(
            intercepted_strategy.len(),
            scherbius_strategy.len(),
            "Intercepted strategy should have the same number of turns as the input strategy."
        );

        // Check that the original encoder in game_state is unchanged
        assert_eq!(
            initial_game_state.encoder.get_rotor_wirings(),
            &original_wirings_before_call, // Compare with the cloned wirings
            "Original game state encoder wirings should not change."
        );
        assert_eq!(
            initial_game_state.encoder.get_steps(),
            &original_steps_before_call, // Compare with the cloned steps
            "Original game state encoder steps should not change."
        );
        // Also, good to ensure the full original encoder is identical if PartialEq is reliable
        assert_eq!(
            &initial_game_state.encoder,
            &original_encoder_clone,
            "The entire original encoder in game_state should be unchanged."
        );


        // Check that the returned encoder has advanced
        // The number of steps taken by the returned_encoder should be equal to the number of cards in the strategy.
        let expected_steps_taken: u32 = scherbius_strategy.iter().map(|cards| cards.len() as u32).sum();

        // We need to calculate the expected final state of the steps vector.
        // This is a bit complex due to odometer stepping.
        // For simplicity, we'll check that steps are different and specifically that the first rotor moved.
        // A more robust check would involve simulating the steps.

        // We know that `call` was invoked scherbius_strategy.len() times on the cloned encoder.
        // Each of those calls could involve multiple `call_char` if cards within a strategy element are processed individually.
        // The `intercept_scherbius_strategy` calls `cloned_encoder.call(h)` for each `h` in `strategy`.
        // The `EasyEnigma::call` method iterates `call_char` for each u32 in its input Vec.
        // So, total `call_char` invocations = sum of lengths of Vecs in `scherbius_strategy`.
        // Total calls to `cloned_encoder.call()` = `scherbius_strategy.len()`.
        // The `EasyEnigma::step` advances after each `call_char`.

        let total_chars_processed = scherbius_strategy.iter().flatten().count();

        if total_chars_processed > 0 {
             assert_ne!(
                returned_encoder.get_steps(),
                &original_steps_before_call,
                "Returned encoder steps should have advanced."
            );
            // If strategy is not empty, the returned encoder should have advanced from its initial state (all zeros or cloned state)
            // The original_encoder_clone was cloned *before* any calls. Its steps are the initial steps.
            // The returned_encoder is a clone that *has* been called. So its steps should be different.
            assert_ne!(
                returned_encoder.get_steps(),
                initial_game_state.encoder.get_steps(), // This is original_steps_before_call
                "Returned encoder steps should be different from the original unchanged steps."
            );

        } else {
            // If strategy is empty, encoder should not have advanced.
            assert_eq!(
                returned_encoder.get_steps(),
                &original_steps_before_call,
                "Returned encoder steps should not have advanced for an empty strategy."
            );
        }

        // Wirings of the returned encoder should be the same as original, only steps change
        assert_eq!(
            returned_encoder.get_rotor_wirings(),
            &original_wirings_before_call,
            "Returned encoder wirings should be the same as original."
        );
    }
}