use std::fmt;
use std::cmp::Ordering;
use std::iter::zip;
use rand::{
    Rng,
    // rngs::ThreadRng // No longer directly used in GameState::step
};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rand::rngs::StdRng;
#[cfg(not(target_arch = "wasm32"))]
use rand::SeedableRng;

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
    // how many values in the encryption code
    pub encryption_code_len: u32,
    // how many values in the encryption vocab
    pub encryption_vocab_size: u32,
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

    encryption: Vec<u32>, // This seems unused, consider removing or implementing its use

    turing_points: u32,
    scherbius_points: u32,

    encoder: enigma::EasyEnigma,
    winner: Actor,

    rng: StdRng, // Consider Box<dyn RngCore + SeedableRng> for more flexibility if needed

    rewards: Vec<Reward>,
    game_config: Arc<GameConfig>,
}

impl GameState {
    pub fn new(game_config: Arc<GameConfig>) -> GameState {
        // On wasm32, StdRng::from_rng might not be ideal.
        // rand::thread_rng() is the standard wasm approach.
        // For consistency, one might use a specific seed source or a simpler RNG for wasm.
        #[cfg(not(target_arch = "wasm32"))]
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        #[cfg(target_arch = "wasm32")]
        let mut rng = StdRng::from_seed(rand::thread_rng().gen::<[u8; 32]>()); // Example for wasm

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
            encryption: utils::sample_random_ints(game_config.encryption_code_len, game_config.encryption_vocab_size, &mut rng),
            encoder: enigma::EasyEnigma::new(game_config.encryption_vocab_size, &mut rng), // Assuming vocab_size for enigma
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
        turing_action: &TuringAction)  -> Result<(), String> {

        // Check that the actions are valid
        self.check_action_validity(&Action::ScherbiusAction(scherbius_action.clone()), &self.scherbius_hand)?;
        self.check_action_validity(&Action::TuringAction(turing_action.clone()), &self.turing_hand)?;

        // Remove cards played from hands
        utils::remove_played_cards_from_hand(&mut self.scherbius_hand, &scherbius_action.strategy);
        utils::remove_played_cards_from_hand(&mut self.turing_hand, &turing_action.strategy);

        // Resolve battles
        let battle_outcomes: Vec<Option<Actor>> = zip(
            scherbius_action.strategy.iter(),
            turing_action.strategy.iter())
                .map(|(s_cards, t_cards)|battle_result(s_cards, t_cards))
                .collect();

        // Distribute the rewards
        for (outcome_option, reward) in zip(battle_outcomes, &self.rewards) {
            if let Some(winner) = outcome_option { // Only distribute reward if there's a winner
                match winner {
                    Actor::Turing => {
                        match reward {
                            Reward::VictoryPoints(v) => self.turing_points += v,
                            Reward::NewCards(cards) => self.turing_hand.extend_from_slice(&cards),
                            Reward::Null => ()
                        }
                    },
                    Actor::Scherbius => {
                        match reward {
                            Reward::VictoryPoints(v) => self.scherbius_points += v,
                            Reward::NewCards(cards) => self.scherbius_hand.extend_from_slice(&cards),
                            Reward::Null => ()
                        }
                    },
                    Actor::Null => () // This case should not be reached if outcome_option is Some.
                }
            }
            // If outcome_option is None (draw), the reward for this battle is not given.
        }

        // Scherbius re-encryption logic
        if scherbius_action.encryption {
            if self.scherbius_points >= self.game_config.encryption_cost {
                self.scherbius_points -= self.game_config.encryption_cost;
                // Assuming encryption_vocab_size is the range for enigma rotor values
                // The original code used hardcoded 0..10. Let's use encryption_vocab_size.
                // enigma::EasyEnigma::set might need adjustment if it expects a fixed size array.
                // For now, assuming it can take dynamically generated values based on vocab_size.
                // If EasyEnigma always uses 2 rotors with values up to vocab_size-1:
                let val1 = self.rng.gen_range(0..self.game_config.encryption_vocab_size);
                let val2 = self.rng.gen_range(0..self.game_config.encryption_vocab_size);
                // This part depends on how `encoder.set` is defined.
                // If it takes `[u32; 2]`, then:
                self.encoder.set([val1, val2]); // Using self.rng for determinism
                self.encoder.reset();
            } else {
                // Optional: Log or handle insufficient points for encryption?
                // For now, if Scherbius can't pay, encryption doesn't happen.
            }
        }

        // Check if a player has won
        if self.scherbius_points >= self.game_config.victory_points {
            self.winner = Actor::Scherbius;
        } else if self.turing_points >= self.game_config.victory_points {
            self.winner = Actor::Turing;
        }

        // If game is over, no need to deal cards or generate new rewards
        if self.winner != Actor::Null {
            return Ok(());
        }

        // Generate rewards for the next round
        let next_rewards = random_rewards(
            self.game_config.n_battles,
            self.game_config.max_vp,
            self.game_config.max_draw,
            &mut self.rng);
        self.rewards = next_rewards;

        // Each player gets some new cards
        let scherbius_new_cards = utils::draw_cards(self.game_config.scherbius_deal, &mut self.rng);
        self.scherbius_hand.extend_from_slice(&scherbius_new_cards);
        let turing_new_cards = utils::draw_cards(self.game_config.turing_deal, &mut self.rng);
        self.turing_hand.extend_from_slice(&turing_new_cards);

        // Clip hands to max_hand_size
        // Note: If players should choose which cards to discard, this logic would be more complex.
        // Simple truncation discards the most recently added cards if over limit.
        // Sorting and then truncating might be another option (e.g., discard lowest value cards).
        // For now, simple truncation:
        if self.scherbius_hand.len() > self.game_config.max_hand_size as usize {
            self.scherbius_hand.truncate(self.game_config.max_hand_size as usize);
        }
        if self.turing_hand.len() > self.game_config.max_hand_size as usize {
            self.turing_hand.truncate(self.game_config.max_hand_size as usize);
        }

        Ok(())
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

#[derive(Debug, Clone)]
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
    encryption_code_len: u32,
    #[pyo3(get)]
    encryption_vocab_size: u32,
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
        encryption_code_len: u32,
        encryption_vocab_size: u32,
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
            encryption_code_len,
            encryption_vocab_size,
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
}

#[pymethods]
impl PyGameState {
    #[new]
    pub fn new(config: Py<PyGameConfig>, py: Python) -> PyResult<Self> {
        // Borrow PyGameConfig from Python to access its fields
        let config_ref = config.borrow(py);

        let game_config = Arc::new(GameConfig {
            scherbius_starting: config_ref.scherbius_starting,
            scherbius_deal: config_ref.scherbius_deal,
            turing_starting: config_ref.turing_starting,
            turing_deal: config_ref.turing_deal,
            victory_points: config_ref.victory_points,
            n_battles: config_ref.n_battles,
            encryption_cost: config_ref.encryption_cost,
            encryption_code_len: config_ref.encryption_code_len,
            encryption_vocab_size: config_ref.encryption_vocab_size,
            verbose: config_ref.verbose,
            max_vp: config_ref.max_vp,
            max_draw: config_ref.max_draw,
            max_hand_size: config_ref.max_hand_size, // New
            max_cards_per_battle: config_ref.max_cards_per_battle, // New
        });

        let game_state = GameState::new(game_config);
        Ok(PyGameState { inner: game_state })
    }

    pub fn step(&mut self,
        turing_strategy: Vec<Cards>,
        scherbius_strategy: Vec<Cards>,
        reencrypt: bool) -> PyResult<()> {

        match self.inner.step(
            &ScherbiusAction { strategy: scherbius_strategy, encryption: reencrypt },
            &TuringAction { strategy: turing_strategy}
        ) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
        }
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
fn turing_vs_scherbius(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<PyGameConfig>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::Arc;

    // Helper to create a default config for tests
    fn create_test_config(n_battles: u32, max_hand_size: u32, max_cards_per_battle: u32) -> Arc<GameConfig> {
        Arc::new(GameConfig {
            scherbius_starting: 5,
            turing_starting: 5,
            scherbius_deal: 2,
            turing_deal: 2,
            victory_points: 10,
            n_battles, // Parameterized
            encryption_cost: 3,
            encryption_code_len: 2, // Corresponds to GameState.encryption, which seems unused
            encryption_vocab_size: 10, // Used for EasyEnigma
            max_vp: 3,
            max_draw: 3,
            verbose: false,
            max_hand_size, // Parameterized
            max_cards_per_battle, // Parameterized
        })
    }

    #[test]
    fn test_game_initialization() {
        let config = create_test_config(3, 10, 5); // n_battles=3, max_hand=10, max_cards_battle=5
        let game = GameState::new(Arc::clone(&config));

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
    fn test_battle_result_logic() { // Renamed for clarity
        assert_eq!(battle_result(&vec![1, 2], &vec![3, 2]), Some(Actor::Turing)); // T wins
        assert_eq!(battle_result(&vec![5, 4], &vec![3, 2]), Some(Actor::Scherbius)); // S wins
        assert_eq!(battle_result(&vec![2, 3], &vec![3, 2]), None); // Draw
    }

    #[test]
    fn test_game_step_basic_flow() { // Renamed for clarity
        let n_battles = 2;
        let max_hand_size = 7;
        let max_cards_per_battle = 3;
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);

        let seed = [0u8; 32]; // For deterministic RNG
        let mut rng = StdRng::from_seed(seed);

        let mut game = GameState {
            turing_hand: vec![1, 2, 3, 4, 5, 6], // 6 cards
            scherbius_hand: vec![10, 20, 30, 40, 50, 60], // 6 cards
            encryption: vec![1, 2], // Unused field, but part of struct
            turing_points: 0,
            scherbius_points: 0,
            encoder: enigma::EasyEnigma::new(config.encryption_vocab_size, &mut rng.clone()), // Clone rng for encoder setup
            winner: Actor::Null,
            rng, // Game uses this rng instance
            rewards: vec![Reward::VictoryPoints(2), Reward::NewCards(vec![100, 101])], // 2 rewards for 2 battles
            game_config: Arc::clone(&config),
        };

        // Scherbius: plays [10], [20,30]. Sums: 10, 50. Hand after play: [40,50,60]
        // Turing: plays [1,2], [3]. Sums: 3, 3. Hand after play: [4,5,6]
        let scherbius_action = ScherbiusAction {
            strategy: vec![vec![10], vec![20, 30]], // Valid: 2 battles, cards per battle <= 3
            encryption: false,
        };
        let turing_action = TuringAction {
            strategy: vec![vec![1, 2], vec![3]], // Valid: 2 battles, cards per battle <= 3
        };

        let step_result = game.step(&scherbius_action, &turing_action);
        assert!(step_result.is_ok(), "Step failed: {:?}", step_result.err());

        // Battle 1: S(10) vs T(3). Scherbius wins. S gets 2 VP. scherbius_points = 2.
        // Battle 2: S(50) vs T(3). Scherbius wins. S gets cards [100, 101].
        // Scherbius hand: [40,50,60] + [100,101] = [40,50,60,100,101] (5 cards)
        // Turing hand: [4,5,6] (3 cards)

        assert_eq!(game.scherbius_points, 2, "Scherbius points mismatch");
        assert_eq!(game.turing_points, 0, "Turing points mismatch");

        // Cards dealt: scherbius_deal=2, turing_deal=2
        // Scherbius hand before truncation: 5 + 2 = 7 cards. Max hand size is 7. No truncation.
        // Turing hand before truncation: 3 + 2 = 5 cards. Max hand size is 7. No truncation.
        // Lengths depend on utils::draw_cards, which is opaque here.
        // We check that the length is AT MOST max_hand_size.
        assert!(game.scherbius_hand.len() <= max_hand_size as usize, "Scherbius hand size exceeds max");
        assert!(game.turing_hand.len() <= max_hand_size as usize, "Turing hand size exceeds max");

        // Example of hand after dealing (assuming draw_cards returns [77,88] for S, [99,111] for T)
        // S_hand: [40,50,60,100,101,S_deal1,S_deal2] -> len 7.
        // T_hand: [4,5,6,T_deal1,T_deal2] -> len 5.
    }

    #[test]
    fn test_max_hand_size_truncation() {
        let n_battles = 1;
        let max_hand_size = 3; // Small max hand size for testing truncation
        let max_cards_per_battle = 1;
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);
        config.clone(); // To avoid unused warning if not used below, though it is.

        let mut game = GameState::new(Arc::clone(&config)); // Starts with 5 cards each, deal 2
        // Initial hands (5 cards) will be truncated after first step if no cards played.
        // Let's set hands manually to test truncation after rewards/dealing.
        game.scherbius_hand = vec![1,2,3,4,5];
        game.turing_hand = vec![1,2,3,4,5];
        game.rewards = vec![Reward::NewCards(vec![100, 101])]; // Turing wins this reward

        let scherbius_action = ScherbiusAction { strategy: vec![vec![1]], encryption: false }; // S plays 1 card
        let turing_action = TuringAction { strategy: vec![vec![2]] }; // T plays 1 card

        // S hand after play: [2,3,4,5] (4 cards)
        // T hand after play: [1,3,4,5] (4 cards)
        // Battle: S(1) vs T(2). Turing wins. T gets [100, 101].
        // T hand after reward: [1,3,4,5,100,101] (6 cards)

        let step_result = game.step(&scherbius_action, &turing_action);
        assert!(step_result.is_ok());

        // After dealing (2 cards each from config.scherbius_deal/turing_deal):
        // S hand: 4 + 2 = 6 cards. Expected truncation to 3.
        // T hand: 6 + 2 = 8 cards. Expected truncation to 3.
        assert_eq!(game.scherbius_hand.len(), max_hand_size as usize, "Scherbius hand not truncated");
        assert_eq!(game.turing_hand.len(), max_hand_size as usize, "Turing hand not truncated");
    }

    #[test]
    fn test_max_cards_per_battle_validation() {
        let n_battles = 1;
        let max_hand_size = 5;
        let max_cards_per_battle = 2; // Max 2 cards for a battle
        let config = create_test_config(n_battles, max_hand_size, max_cards_per_battle);

        let mut game = GameState::new(Arc::clone(&config));
        game.scherbius_hand = vec![1,2,3,4,5];
        game.turing_hand = vec![1,2,3,4,5];

        // Valid action
        let valid_s_action = ScherbiusAction { strategy: vec![vec![1,2]], encryption: false };
        let valid_t_action = TuringAction { strategy: vec![vec![3,4]] };
        let result = game.check_action_validity(&Action::ScherbiusAction(valid_s_action.clone()), &game.scherbius_hand);
        assert!(result.is_ok());
        let result = game.check_action_validity(&Action::TuringAction(valid_t_action.clone()), &game.turing_hand);
        assert!(result.is_ok());


        // Invalid action: too many cards in one battle
        let invalid_s_action = ScherbiusAction { strategy: vec![vec![1,2,3]], encryption: false }; // 3 cards > max 2
        let result = game.check_action_validity(&Action::ScherbiusAction(invalid_s_action), &game.scherbius_hand);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Too many cards committed"));
    }

    #[test]
    fn test_encryption_cost() {
        let config = create_test_config(1, 5, 2);
        let mut game = GameState::new(Arc::clone(&config));
        game.scherbius_points = 5; // Enough points to pay for encryption (cost 3)
        game.scherbius_hand = vec![1,2,3]; // Ensure valid hand for strategy
        game.turing_hand = vec![1,2,3];   // Ensure valid hand for strategy
        game.rewards = vec![Reward::VictoryPoints(1)];


        let scherbius_action = ScherbiusAction { strategy: vec![vec![1]], encryption: true };
        let turing_action = TuringAction { strategy: vec![vec![2]] }; // Turing wins, gets 1 VP

        let initial_s_points = game.scherbius_points;
        let encryption_cost = game.game_config.encryption_cost;

        let step_result = game.step(&scherbius_action, &turing_action);
        assert!(step_result.is_ok());

        // Scherbius should pay encryption_cost. Battle S(1) vs T(2), T wins. S gets 0 VP from battle.
        assert_eq!(game.scherbius_points, initial_s_points - encryption_cost);
        assert_eq!(game.turing_points, 1); // Turing wins the battle reward
    }

     #[test]
    fn test_strategy_must_cover_all_battles() {
        let n_battles = 2; // Game expects 2 battles
        let config = create_test_config(n_battles, 5, 2);
        let game = GameState::new(Arc::clone(&config));

        // Action with only 1 battle strategy
        let scherbius_action_not_enough_battles = ScherbiusAction {
            strategy: vec![vec![1]], // Only 1 battle, expected 2
            encryption: false,
        };
        let result = game.check_action_validity(
            &Action::ScherbiusAction(scherbius_action_not_enough_battles),
            &game.scherbius_hand
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Strategy must cover all battles"));
    }
}