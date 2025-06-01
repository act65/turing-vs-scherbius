
use std::cmp::Ordering;
use std::iter::zip;
use rand::{
    Rng,
    SeedableRng, // Ensure SeedableRng is imported
    rngs::StdRng // Make StdRng available unconditionally
    // rngs::ThreadRng // No longer directly used in GameState::step
};
use std::sync::Arc;

// For wasm32, StdRng might not be available or ideal.
// Consider using a wasm-friendly RNG if targeting wasm32 primarily for GameState's internal RNG.
// For now, we'll assume StdRng is acceptable or this part is non-wasm critical.
// If GameState needs to be fully wasm32 compatible, its rng field might need to be RngCore + SeedableRng.
// However, rand::thread_rng() is used for wasm, and StdRng::from_rng can take that.

pub mod enigma;
pub mod utils;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GameConfig {
    // how many cards each player starts with
    pub scherbius_starting: u32,
    pub turing_starting: u32,
    // how many cards each player gets each round
    pub scherbius_deal: u32,
    pub turing_deal: u32,
    // how many victory points required to win
    pub victory_points: u32,
    // how many battles each round
    pub n_battles: u32,
    // re-encrypting costs victory points
    pub encryption_cost: u32,
    // how many values in the encryption vocab
    pub encryption_vocab_size: u32,
    // how many rotors to use
    pub encryption_k_rotors: u32,
    // maximum value of victory points in the rewards
    pub max_vp: u32,
    // maximum number of cards in the rewards
    pub max_draw: u32,
    pub verbose: bool,
    // max hand size
    pub max_hand_size: u32,
    // Max cards one player can commit to a single battle
    pub max_cards_per_battle: u32,
}

#[derive(Debug, Clone)]
struct GameState {
    turing_hand: Vec<u32>,
    scherbius_hand: Vec<u32>,

    turing_points: u32,
    scherbius_points: u32,

    encoder: enigma::EasyEnigma,
    winner: Actor,

    rng: StdRng, // Consider Box<dyn RngCore + SeedableRng> for more flexibility if needed

    rewards: Vec<Reward>,
    game_config: Arc<GameConfig>,
}


// Add a new struct to hold detailed battle results
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleOutcomeDetail {
    #[pyo3(get)] // Expose fields to Python
    turing_sum: u32,
    #[pyo3(get)]
    scherbius_sum: u32,
    #[pyo3(get)]
    turing_cards_won: Vec<u32>,
    #[pyo3(get)]
    turing_vp_won: u32,
}

#[pymethods] // Add pymethods if you need to create it from Python (optional for now)
impl BattleOutcomeDetail {
    // No #[new] needed if it's only created in Rust and passed to Python
}


impl GameState {
    pub fn new(game_config: Arc<GameConfig>, seed: Option<u64>) -> GameState {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                #[cfg(not(target_arch = "wasm32"))]
                { StdRng::from_rng(rand::thread_rng()).unwrap() }
                #[cfg(target_arch = "wasm32")]
                { StdRng::from_seed(rand::thread_rng().gen::<[u8; 32]>()) }
            }
        };

        let rewards = random_rewards(
            game_config.n_battles,
            game_config.max_vp,
            game_config.max_draw,
            &mut rng);

        GameState{
            scherbius_hand: utils::draw_cards(game_config.scherbius_starting, &mut rng),
            turing_hand: utils::draw_cards(game_config.turing_starting, &mut rng),
            turing_points: 0,
            scherbius_points: 0,
            // encryption field is initialized but not clearly used later.
            // If it's for the game's core encryption concept, it should be integrated.
            // For now, assuming it's separate from enigma::EasyEnigma's internal state.
            encoder: enigma::EasyEnigma::new(game_config.encryption_vocab_size, 
                game_config.encryption_k_rotors as usize,
                &mut rng), // Assuming vocab_size for enigma
            winner: Actor::Null,
            rng: rng,
            rewards: rewards,
            game_config: Arc::clone(&game_config), // Moved to end for conventional order
        }
    }

    fn intercept_scherbius_strategy(&mut self, strategy: &[Cards]) -> Vec<Cards> {
        strategy.iter().map(|h| self.encoder.call(h)).collect()
    }

    fn check_action_validity(&self, action: &Action, hand: &Vec<u32>) -> Result<(), String> {
        let (player_name, current_strategy) = match action {
            Action::TuringAction(turing_action) => ("Turing", &turing_action.strategy),
            Action::ScherbiusAction(scherbius_action) => ("Scherbius", &scherbius_action.strategy),
        };

        if current_strategy.len() as u32 != self.game_config.n_battles {
            return Err(format!(
                "{}: Strategy must cover all battles. Expected: {}, Got: {}",
                player_name,
                self.game_config.n_battles,
                current_strategy.len()
            ));
        }

        if !utils::is_subset_of_hand(current_strategy, hand) {
            return Err(format!("{}: Strategy is not subset of hand", player_name));
        }

        for cards_in_battle in current_strategy {
            if cards_in_battle.len() as u32 > self.game_config.max_cards_per_battle {
                return Err(format!(
                    "{}: Too many cards committed to a single battle. Max: {}, Got: {}",
                    player_name,
                    self.game_config.max_cards_per_battle,
                    cards_in_battle.len()
                ));
            }
        }
        Ok(())
    }

        fn step(
        &mut self,
        scherbius_action: &ScherbiusAction,
        turing_action: &TuringAction,
    ) -> Result<Vec<BattleOutcomeDetail>, String> { // Return type is fine

        self.check_action_validity(&Action::ScherbiusAction(scherbius_action.clone()), &self.scherbius_hand)?;
        self.check_action_validity(&Action::TuringAction(turing_action.clone()), &self.turing_hand)?;

        utils::remove_played_cards_from_hand(&mut self.scherbius_hand, &scherbius_action.strategy);
        utils::remove_played_cards_from_hand(&mut self.turing_hand, &turing_action.strategy);

        let mut detailed_outcomes: Vec<BattleOutcomeDetail> = Vec::new();

        for (i, (s_cards, t_cards)) in zip(
            scherbius_action.strategy.iter(),
            turing_action.strategy.iter(),
        ).enumerate() {
            let t_sum = t_cards.iter().sum::<u32>();
            let s_sum = s_cards.iter().sum::<u32>();
            let winner_option = battle_result(s_cards, t_cards);

            let mut turing_cards_won_in_battle = Vec::new();
            let mut turing_vp_won_in_battle = 0;

            if let Some(winner) = winner_option {
                let reward_for_battle = &self.rewards[i];
                match winner {
                    Actor::Turing => {
                        match reward_for_battle {
                            Reward::VictoryPoints(v) => {
                                self.turing_points += v;
                                turing_vp_won_in_battle = *v;
                            }
                            Reward::NewCards(cards) => {
                                self.turing_hand.extend_from_slice(&cards);
                                turing_cards_won_in_battle = cards.clone();
                            }
                            Reward::Null => (),
                        }
                    }
                    Actor::Scherbius => {
                        match reward_for_battle {
                            Reward::VictoryPoints(v) => self.scherbius_points += v,
                            Reward::NewCards(cards) => self.scherbius_hand.extend_from_slice(&cards),
                            Reward::Null => (),
                        }
                    }
                    Actor::Null => (),
                }
            }
            detailed_outcomes.push(BattleOutcomeDetail { // This is fine, creating Rust struct
                turing_sum: t_sum,
                scherbius_sum: s_sum,
                turing_cards_won: turing_cards_won_in_battle,
                turing_vp_won: turing_vp_won_in_battle,
            });
        }


        if scherbius_action.encryption {
            if self.scherbius_points >= self.game_config.encryption_cost {
                self.scherbius_points -= self.game_config.encryption_cost;
                // Create a new EasyEnigma instance using the game's RNG
                // This effectively changes all wirings and resets steps to [0,0]
                self.encoder = enigma::EasyEnigma::new(
                    self.game_config.encryption_vocab_size, 
                    self.game_config.encryption_k_rotors as usize,
                    &mut self.rng);
                // No need for self.encoder.reset() as new() initializes steps to [0,0].
                // No need for self.encoder.set(...) as rotor values are now complex wirings.
                if self.game_config.verbose {
                    println!("Scherbius re-encrypted. New Enigma settings generated.");
                }
            } else {
                 if self.game_config.verbose {
                    println!("Scherbius attempted re-encryption but lacked points.");
                }
            }
        }

        if self.scherbius_points >= self.game_config.victory_points {
            self.winner = Actor::Scherbius;
        } else if self.turing_points >= self.game_config.victory_points {
            self.winner = Actor::Turing;
        }

        if self.winner != Actor::Null {
            return Ok(detailed_outcomes);
        }

        let next_rewards = random_rewards(
            self.game_config.n_battles,
            self.game_config.max_vp,
            self.game_config.max_draw,
            &mut self.rng,
        );
        self.rewards = next_rewards;

        let scherbius_new_cards = utils::draw_cards(self.game_config.scherbius_deal, &mut self.rng);
        self.scherbius_hand.extend_from_slice(&scherbius_new_cards);
        let turing_new_cards = utils::draw_cards(self.game_config.turing_deal, &mut self.rng);
        self.turing_hand.extend_from_slice(&turing_new_cards);

        if self.scherbius_hand.len() > self.game_config.max_hand_size as usize {
            self.scherbius_hand.truncate(self.game_config.max_hand_size as usize);
        }
        if self.turing_hand.len() > self.game_config.max_hand_size as usize {
            self.turing_hand.truncate(self.game_config.max_hand_size as usize);
        }

        // reset enigma steps
        self.encoder.reset_steps();

        Ok(detailed_outcomes)
    }
}

pub type ScherbiusPlayer = fn(&Vec<u32>, &Vec<Reward>) -> ScherbiusAction;
pub type TuringPlayer = fn(&Vec<u32>, &Vec<Reward>, &Vec<Cards>) -> TuringAction;

pub type Cards = Vec<u32>;
pub type EncryptionCode = Vec<u32>; // This type seems unused, like GameState.encryption

#[derive(Debug, PartialEq, Clone, Copy)] // Added Copy
pub enum Actor {
    Scherbius,
    Turing,
    Null
}


fn battle_result(
    scherbius_cards: &Cards,
    turing_cards: &Cards) -> Option<Actor> {
    match scherbius_cards.iter().sum::<u32>().cmp(&turing_cards.iter().sum::<u32>()) {
        Ordering::Less => Some(Actor::Turing),
        Ordering::Greater => Some(Actor::Scherbius),
        Ordering::Equal => None,
    }
}

#[derive(Debug, Clone, PartialEq)] // Added PartialEq
pub enum Reward {
    VictoryPoints(u32),
    NewCards(Vec<u32>),
    Null,
}

fn sample_battle_reward(max_vp: u32, max_draw: u32, rng: &mut StdRng) -> Reward {
    match rng.gen_range(0..2) { // 0 for VP, 1 for NewCards
        0 => {
            if max_vp == 0 {
                Reward::VictoryPoints(0) // Or Reward::Null if 0 VP means no reward of this type
            } else {
                // Using uniform distribution. Max value is inclusive.
                Reward::VictoryPoints(rng.gen_range(1..=max_vp))
            }
        }
        1 => {
            if max_draw == 0 {
                Reward::NewCards(Vec::new()) // Or Reward::Null
            } else {
                let num_cards = rng.gen_range(1..=max_draw);
                Reward::NewCards(utils::draw_cards(num_cards, rng))
            }
        }
        _ => unreachable!("rng.gen_range(0..2) should only produce 0 or 1"),
    }
}

fn random_rewards(n: u32, max_vp: u32, max_draw: u32, rng: &mut StdRng)->Vec<Reward> {
    (0..n).map(|_| sample_battle_reward(max_vp, max_draw, rng)).collect()
}

#[derive(Debug, Clone)]
pub struct TuringAction {
    pub strategy: Vec<Cards>
}

#[derive(Debug, Clone)]
pub struct ScherbiusAction {
    pub strategy: Vec<Cards>,
    pub encryption: bool,
}

#[derive(Debug, Clone)]
pub enum Action {
    TuringAction(TuringAction),
    ScherbiusAction(ScherbiusAction),
}

// check_action_validity is now a method of GameState

use pyo3::prelude::*;
// use pyo3::wrap_pyfunction; // Not used

#[pyclass]
#[derive(Clone)]
pub struct PyGameConfig {
    #[pyo3(get)]
    scherbius_starting: u32,
    #[pyo3(get)]
    scherbius_deal: u32,
    #[pyo3(get)]
    turing_starting: u32,
    #[pyo3(get)]
    turing_deal: u32,
    #[pyo3(get)]
    victory_points: u32,
    #[pyo3(get)]
    n_battles: u32,
    #[pyo3(get)]
    encryption_cost: u32,
    #[pyo3(get)]
    encryption_vocab_size: u32,
    #[pyo3(get)]
    encryption_k_rotors: u32,
    #[pyo3(get)]
    verbose: bool,
    #[pyo3(get)]
    max_vp: u32,
    #[pyo3(get)]
    max_draw: u32,
    #[pyo3(get)]
    max_hand_size: u32, // New
    #[pyo3(get)]
    max_cards_per_battle: u32, // New
}

#[pymethods]
impl PyGameConfig {
    #[new]
    #[allow(clippy::too_many_arguments)] // Common for config constructors
    pub fn new(
        scherbius_starting: u32,
        scherbius_deal: u32,
        turing_starting: u32,
        turing_deal: u32,
        victory_points: u32,
        n_battles: u32,
        encryption_cost: u32,
        encryption_vocab_size: u32,
        encryption_k_rotors: u32,
        verbose: bool,
        max_vp: u32,
        max_draw: u32,
        max_hand_size: u32, // New
        max_cards_per_battle: u32, // New
    ) -> Self {
        PyGameConfig {
            scherbius_starting,
            scherbius_deal,
            turing_starting,
            turing_deal,
            victory_points,
            n_battles,
            encryption_cost,
            encryption_vocab_size,
            encryption_k_rotors,
            verbose,
            max_vp,
            max_draw,
            max_hand_size, // New
            max_cards_per_battle, // New
        }
    }
}


#[pyclass]
struct PyGameState {
    inner: GameState,
    // This field will now store Py<BattleOutcomeDetail> or similar if you want Python objects directly.
    // Or, keep it as Vec<BattleOutcomeDetail> and convert on the fly in battle_results.
    // Keeping as Vec<BattleOutcomeDetail> is simpler for Rust-internal storage.
    last_battle_outcomes: Vec<BattleOutcomeDetail>,
}

#[pymethods]
impl PyGameState {
    #[new]
    pub fn new(config: Py<PyGameConfig>, seed: Option<u64>, py: Python) -> PyResult<Self> {
        let config_ref = config.borrow(py);
        let game_config = Arc::new(GameConfig {
            scherbius_starting: config_ref.scherbius_starting,
            scherbius_deal: config_ref.scherbius_deal,
            turing_starting: config_ref.turing_starting,
            turing_deal: config_ref.turing_deal,
            victory_points: config_ref.victory_points,
            n_battles: config_ref.n_battles,
            encryption_cost: config_ref.encryption_cost,
            encryption_vocab_size: config_ref.encryption_vocab_size,
            encryption_k_rotors: config_ref.encryption_k_rotors,
            verbose: config_ref.verbose,
            max_vp: config_ref.max_vp,
            max_draw: config_ref.max_draw,
            max_hand_size: config_ref.max_hand_size,
            max_cards_per_battle: config_ref.max_cards_per_battle,
        });

        let game_state = GameState::new(game_config, seed);
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
        match self.inner.step(
            &ScherbiusAction {
                strategy: scherbius_strategy,
                encryption: reencrypt,
            },
            &TuringAction {
                strategy: turing_strategy,
            },
        ) {
            Ok(outcomes) => {
                self.last_battle_outcomes = outcomes;
                Ok(())
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }

    // New method to expose the battle results
    // Returns a Python list of BattleOutcomeDetail objects
    pub fn battle_results(&self, py: Python) -> PyResult<Vec<Py<BattleOutcomeDetail>>> {
        // The `py: Python` argument gives us the GIL token.
        let mut py_outcomes = Vec::new();
        for outcome_detail in &self.last_battle_outcomes {
            // Create a Python instance of BattleOutcomeDetail
            // Py::new takes a Python token and the Rust struct.
            // The Rust struct BattleOutcomeDetail must be #[pyclass] for this to work.
            let py_outcome_detail = Py::new(py, outcome_detail.clone())?;
            py_outcomes.push(py_outcome_detail);
        }
        Ok(py_outcomes)
    }

    pub fn is_won(&self) -> bool {
        self.inner.winner != Actor::Null
    }

    pub fn turing_points(&self) -> u32 {
        self.inner.turing_points
    }

    pub fn scherbius_points(&self) -> u32 {
        self.inner.scherbius_points
    }

    pub fn turing_hand(&self) -> Vec<u32> {
        self.inner.turing_hand.clone()
    }

    pub fn scherbius_hand(&self) -> Vec<u32> {
        self.inner.scherbius_hand.clone()
    }

    pub fn winner(&self) -> &str {
        match self.inner.winner {
            Actor::Scherbius => "Scherbius",
            Actor::Turing => "Turing",
            Actor::Null => "Null",
        }
    }

    pub fn rewards(&self) -> (Vec<Vec<u32>>, Vec<u32>) {
        let rewards = self.inner.rewards.clone();
        let mut new_cards_rewards: Vec<Vec<u32>> = Vec::new();
        let mut victory_points_rewards: Vec<u32> = Vec::new();

        for reward in rewards {
            match reward {
                Reward::VictoryPoints(v) => {
                    victory_points_rewards.push(v);
                    new_cards_rewards.push(vec![]); // Placeholder for consistent length
                },
                Reward::NewCards(cards) => {
                    new_cards_rewards.push(cards);
                    victory_points_rewards.push(0); // Placeholder
                },
                Reward::Null => {
                    new_cards_rewards.push(vec![]);
                    victory_points_rewards.push(0);
                }
            }
        }
        (new_cards_rewards, victory_points_rewards)
    }

    pub fn turing_observation(&mut self, scherbius_strategy: Vec<Vec<u32>>) -> (Vec<u32>, Vec<Vec<u32>>) {
        let intercepted_scherbius_strategy = self.inner.intercept_scherbius_strategy(&scherbius_strategy);
        (self.inner.turing_hand.clone(), intercepted_scherbius_strategy)
    }

    pub fn scherbius_observation(&self) -> Vec<u32> {
        self.inner.scherbius_hand.clone()
    }
}

#[pymodule]
fn tvs_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<PyGameConfig>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    // use rand::rngs::StdRng; // Already imported at top level
    // use rand::SeedableRng; // Already imported at top level
    // use std::sync::Arc; // Already imported at top level

    // Helper to create a default config for tests
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
        assert_eq!(battle_result(&vec![1, 2], &vec![3, 2]), Some(Actor::Turing)); 
        assert_eq!(battle_result(&vec![5, 4], &vec![3, 2]), Some(Actor::Scherbius)); 
        assert_eq!(battle_result(&vec![2, 3], &vec![3, 2]), None); 
    }

    #[test]
    fn test_game_step_basic_flow_and_outcomes() { // Updated test name
        let n_battles = 2;
        let max_hand_size = 7;
        let max_cards_per_battle = 3;
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);

        let seed = [0u8; 32]; 
        let rng = StdRng::from_seed(seed);

        let mut game = GameState {
            turing_hand: vec![1, 2, 3, 4, 5, 6], 
            scherbius_hand: vec![10, 20, 30, 40, 50, 60], 
            turing_points: 0,
            scherbius_points: 0,
            encoder: enigma::EasyEnigma::new(
                config.encryption_vocab_size, 
                config.encryption_k_rotors as usize,
                &mut rng.clone()), 
            winner: Actor::Null,
            rng, 
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

        let step_result = game.step(&scherbius_action, &turing_action);
        assert!(step_result.is_ok(), "Step failed: {:?}", step_result.as_ref().err());
        let outcomes = step_result.unwrap();

        assert_eq!(outcomes.len(), 2);
        // Battle 1: S(10) vs T(3). Scherbius wins. S gets 2 VP.
        assert_eq!(outcomes[0].scherbius_sum, 10);
        assert_eq!(outcomes[0].turing_sum, 3);
        assert_eq!(outcomes[0].turing_vp_won, 0);
        assert_eq!(outcomes[0].turing_cards_won.len(), 0);

        // Battle 2: S(50) vs T(3). Scherbius wins. S gets cards [100, 101].
        assert_eq!(outcomes[1].scherbius_sum, 50);
        assert_eq!(outcomes[1].turing_sum, 3);
        assert_eq!(outcomes[1].turing_vp_won, 0);
        assert_eq!(outcomes[1].turing_cards_won.len(), 0); // Scherbius won cards, not Turing

        assert_eq!(game.scherbius_points, 2, "Scherbius points mismatch");
        assert_eq!(game.turing_points, 0, "Turing points mismatch");

        assert!(game.scherbius_hand.len() <= max_hand_size as usize, "Scherbius hand size exceeds max");
        assert!(game.turing_hand.len() <= max_hand_size as usize, "Turing hand size exceeds max");
    }

    #[test]
    fn test_max_hand_size_truncation() {
        let n_battles = 1;
        let max_hand_size = 3; 
        let max_cards_per_battle = 1;
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);
        
        let mut game = GameState::new(Arc::clone(&config), None); 
        game.scherbius_hand = vec![1,2,3,4,5];
        game.turing_hand = vec![1,2,3,4,5];
        game.rewards = vec![Reward::NewCards(vec![100, 101])]; 

        let scherbius_action = ScherbiusAction { strategy: vec![vec![1]], encryption: false }; 
        let turing_action = TuringAction { strategy: vec![vec![2]] }; 

        let step_result = game.step(&scherbius_action, &turing_action);
        assert!(step_result.is_ok());

        assert_eq!(game.scherbius_hand.len(), max_hand_size as usize, "Scherbius hand not truncated");
        assert_eq!(game.turing_hand.len(), max_hand_size as usize, "Turing hand not truncated");
    }

    #[test]
    fn test_max_cards_per_battle_validation() {
        let n_battles = 1;
        let max_hand_size = 5;
        let max_cards_per_battle = 2; 
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);

        let mut game = GameState::new(Arc::clone(&config), None); 
        game.scherbius_hand = vec![1,2,3,4,5];
        game.turing_hand = vec![1,2,3,4,5];

        let valid_s_action = ScherbiusAction { strategy: vec![vec![1,2]], encryption: false };
        let valid_t_action = TuringAction { strategy: vec![vec![3,4]] };
        let result = game.check_action_validity(&Action::ScherbiusAction(valid_s_action.clone()), &game.scherbius_hand);
        assert!(result.is_ok());
        let result = game.check_action_validity(&Action::TuringAction(valid_t_action.clone()), &game.turing_hand);
        assert!(result.is_ok());

        let invalid_s_action = ScherbiusAction { strategy: vec![vec![1,2,3]], encryption: false }; 
        let result = game.check_action_validity(&Action::ScherbiusAction(invalid_s_action), &game.scherbius_hand);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Too many cards committed"));
    }

    #[test]
    fn test_encryption_cost() {
        let config = create_test_config(1, 5, 2);
        let mut game = GameState::new(Arc::clone(&config), None); 
        game.scherbius_points = 5; 
        game.scherbius_hand = vec![1,2,3]; 
        game.turing_hand = vec![1,2,3];   
        game.rewards = vec![Reward::VictoryPoints(1)];


        let scherbius_action = ScherbiusAction { strategy: vec![vec![1]], encryption: true };
        let turing_action = TuringAction { strategy: vec![vec![2]] }; 

        let initial_s_points = game.scherbius_points;
        let encryption_cost = game.game_config.encryption_cost;

        let step_result = game.step(&scherbius_action, &turing_action);
        assert!(step_result.is_ok());

        assert_eq!(game.scherbius_points, initial_s_points - encryption_cost);
        assert_eq!(game.turing_points, 1); 
    }

     #[test]
    fn test_strategy_must_cover_all_battles() {
        let n_battles = 2; 
        let config = create_test_config(n_battles, 5, 2);
        let game = GameState::new(Arc::clone(&config), None); 

        let scherbius_action_not_enough_battles = ScherbiusAction {
            strategy: vec![vec![1]], 
            encryption: false,
        };
        let result = game.check_action_validity(
            &Action::ScherbiusAction(scherbius_action_not_enough_battles),
            &game.scherbius_hand
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Strategy must cover all battles"));
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

        assert_ne!(game1.turing_hand, game2.turing_hand, "Turing hands should differ (highly probable)");
        assert_ne!(game1.scherbius_hand, game2.scherbius_hand, "Scherbius hands should differ (highly probable)");

        if game1.rewards.len() == game2.rewards.len() {
            let rewards_match = game1.rewards.iter().zip(game2.rewards.iter()).all(|(r1, r2)| r1 == r2);
            assert!(!rewards_match, "Rewards lists should differ (highly probable)");
        } 
        
        assert_ne!(game1.encoder.get_rotor_wirings(), game2.encoder.get_rotor_wirings(), "Enigma rotors should differ (highly probable)");
    }

    #[test]
    fn test_none_seed_initializes_gamestate() {
        let config = create_test_config(3, 10, 5);
        let game = GameState::new(Arc::clone(&config), None); 

        assert_eq!(game.turing_hand.len(), config.turing_starting as usize, "Turing hand size mismatch");
        assert_eq!(game.scherbius_hand.len(), config.scherbius_starting as usize, "Scherbius hand size mismatch");
        assert_eq!(game.rewards.len(), config.n_battles as usize, "Rewards count mismatch");
        assert_eq!(game.turing_points, 0, "Initial Turing points should be 0");
        assert_eq!(game.scherbius_points, 0, "Initial Scherbius points should be 0");
        assert_eq!(game.winner, Actor::Null, "Initial winner should be Null");
    }
}