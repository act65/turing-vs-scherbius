use std::fmt;
use std::cmp::Ordering;
use std::iter::zip;
use rand::{
    Rng,
    rngs::ThreadRng
};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rand::rngs::StdRng;
#[cfg(not(target_arch = "wasm32"))]
use rand::SeedableRng;

pub mod enigma;
pub mod utils;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GameConfig {
    pub scherbius_starting: u32,
    pub scherbius_deal: u32,
    pub turing_starting: u32,
    pub turing_deal: u32,
    pub victory_points: u32, // how many victory points required to win
    pub n_battles: u32,
    pub encryption_cost: u32, // re-encrypting costs victory points
    pub encryption_code_len: u32,
    pub encryption_vocab_size: u32,
    pub verbose: bool,
    pub max_vp: u32,
    pub max_draw: u32,
}

#[derive(Debug, Clone)]
struct GameState {
    turing_hand: Vec<u32>,
    scherbius_hand: Vec<u32>,

    encryption_broken: bool,
    encryption: Vec<u32>,
    // TODO want to vary the number of values used for encryption?!
    // 10^2 = 100. quite hard to break encryption!

    turing_points: u32,
    scherbius_points: u32,

    encoder: enigma::EasyEnigma,
    winner: Actor,

    rng: StdRng,

    rewards: Vec<Reward>,
    game_config: Arc<GameConfig>,
}

// impl fmt::Display for GameState {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "Turing: {:?}\nSherbius: {:?}", self.turing_hand, self.scherbius_hand)
//     }
// }

impl GameState {
    pub fn new(game_config: Arc<GameConfig>) -> GameState {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        let rewards = random_rewards(
            game_config.n_battles,
            game_config.max_vp, 
            game_config.max_draw,
            &mut rng);
        
        GameState{
            game_config: Arc::clone(&game_config),
            scherbius_hand: utils::draw_cards(game_config.scherbius_starting, &mut rng),
            turing_hand: utils::draw_cards(game_config.turing_starting, &mut rng),
            turing_points: 0,
            scherbius_points: 0,
            encryption_broken: false,
            encryption: utils::sample_random_ints(game_config.encryption_code_len, game_config.encryption_vocab_size, &mut rng),
            encoder: enigma::EasyEnigma::new(10, &mut rng),
            winner: Actor::Null,
            rng: rng,
            rewards: rewards,
        }
    }

    fn intercept_scherbius_strategy(&mut self, strategy: &Vec<Cards>) -> Vec<Cards> {

        if self.encryption_broken {
            return strategy.clone()}  // probs dont need to clone?
        else {
            let mut encrypted_strategy: Vec<Cards> = Vec::new();
            for h in strategy {
                encrypted_strategy.push(self.encoder.call(&h));
            }
            return encrypted_strategy}
    }

    fn step(
        &mut self,
        scherbius_action: &ScherbiusAction,
        turing_action: &TuringAction) {

    // check that the actions are valid
    check_action_validity(&Action::ScherbiusAction(scherbius_action.clone()), &self.scherbius_hand);
    check_action_validity(&Action::TuringAction(turing_action.clone()), &self.turing_hand);

    // each player gets some new cards
    let new_cards = utils::draw_cards(self.game_config.scherbius_deal, &mut self.rng);
    self.scherbius_hand.extend_from_slice(&new_cards);
    let new_cards = utils::draw_cards(self.game_config.turing_deal, &mut self.rng);
    self.turing_hand.extend_from_slice(&new_cards);

    // remove cards played from hands
    utils::remove_played_cards_from_hand(&mut self.scherbius_hand, &scherbius_action.strategy);
    utils::remove_played_cards_from_hand(&mut self.turing_hand, &turing_action.strategy);
    utils::remove_played_cards_from_hand(&mut self.turing_hand, &turing_action.guesses);
    // TODO missing removing victory pts for re-encryption?!

    // resolve battles
    let results: Vec<_> = zip(
        scherbius_action.strategy.iter(), 
        turing_action.strategy.iter())
            .map(|(a1, a2)|battle_result(a1, a2))
            // this shouldnt work?!
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .collect();

    // TODO test that;
    // - draw means no one wins
    // - no cards vs has cards means player with cards wins

    // distribute the rewards
    for (result, reward) in zip(results, &self.rewards) {
        match result {
            Actor::Turing => 
                {match reward {
                    Reward::VictoryPoints(v) => self.turing_points = self.turing_points + v,
                    Reward::NewCards(cards) => self.turing_hand.extend_from_slice(&cards),
                    _ => ()
                    }},
            Actor::Scherbius => 
                {match reward {
                    Reward::VictoryPoints(v) => self.scherbius_points = self.scherbius_points + v,
                    Reward::NewCards(cards) => self.scherbius_hand.extend_from_slice(&cards),
                    _ => ()
                }}
            _ => ()
        }

    }

    // resolve encryption guess
    for g in turing_action.guesses.iter() {
        if g == &self.encryption {self.encryption_broken=true}
    }

    // reset encryption?
    let mut rng = rand::thread_rng();
    if self.scherbius_points >= self.game_config.encryption_cost && scherbius_action.encryption 
        {self.encoder.set([rng.gen_range(0..10), rng.gen_range(0..10)]);
        self.encoder.reset();
        self.encryption_broken=false};

    // check if a player has won
    if self.scherbius_points >= self.game_config.victory_points {
        self.winner = Actor::Scherbius;}
    else if self.turing_points >= self.game_config.victory_points {
        self.winner = Actor::Turing;
    }

    // what is being played for next round?
    let rewards = random_rewards(
        self.game_config.n_battles,
        self.game_config.max_vp, 
        self.game_config.max_draw,
        &mut self.rng);
    
    self.rewards = rewards;

    }
}

pub type ScherbiusPlayer = fn(&Vec<u32>, &Vec<Reward>) -> ScherbiusAction;
pub type TuringPlayer = fn(&Vec<u32>, &Vec<Reward>, &Vec<Cards>) -> TuringAction;

pub type Cards = Vec<u32>;
pub type EncryptionCode = Vec<u32>;

#[derive(Debug, PartialEq, Clone)]
pub enum Actor {
    Scherbius,
    Turing,
    Null
}


fn battle_result(
    scherbius_cards: &Cards, 
    turing_cards: &Cards) -> Option<Actor> {
    match (scherbius_cards.iter().sum::<u32>()).cmp(&turing_cards.iter().sum::<u32>()) {
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
    match rng.gen_range(0..2) {
        // TODO distribution should make larger values less likely
        // TODO want a parameter to control the max value in GameConfig?!
        // TODO move rng to be an arg?
        0 => Reward::VictoryPoints(rng.gen_range(1..max_vp)),
        1 => Reward::NewCards(utils::draw_cards(rng.gen_range(1..max_draw), rng)),
        _ => Reward::Null,
    }
}

fn random_rewards(n: u32, max_vp: u32, max_draw: u32, rng: &mut StdRng)->Vec<Reward> {
    (0..n).map(|_| sample_battle_reward(max_vp, max_draw, rng)).collect()
}

#[derive(Debug, Clone)]
pub struct TuringAction {
    pub strategy: Vec<Cards>,
    pub guesses: Vec<EncryptionCode>,
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

fn check_action_validity(action: &Action, hand: &Vec<u32>) {
    match action {
        Action::TuringAction(turing_action) => {
            if !utils::is_subset_of_hand(&turing_action.strategy, hand) {
                panic!("Strategy is not subset of hand")
            }
            if !utils::is_subset_of_hand(&turing_action.guesses, hand) {
                panic!("Guesses is not subset of hand")
            }
        }
        Action::ScherbiusAction(scherbius_action) => {
            if !utils::is_subset_of_hand(&scherbius_action.strategy, hand) {
                panic!("Strategy is not subset of hand")
            }
        }
    }
}


use pyo3::prelude::*;
use pyo3::wrap_pyfunction;


#[pyclass]
#[derive(Clone)]
pub struct PyGameConfig {
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
    max_draw: u32
}

#[pymethods]
impl PyGameConfig {
    #[new]
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
        max_draw: u32) -> Self {
        PyGameConfig {
            scherbius_starting: scherbius_starting,
            scherbius_deal: scherbius_deal,
            turing_starting: turing_starting,
            turing_deal: turing_deal,
            victory_points: victory_points,
            n_battles: n_battles,
            encryption_cost: encryption_cost,
            encryption_code_len: encryption_code_len,
            encryption_vocab_size: encryption_vocab_size,
            verbose: verbose,
            max_vp: max_vp,
            max_draw: max_draw}
        }
}

#[pyclass]
struct PyGameState {
    inner: GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    pub fn new(config: PyGameConfig) -> PyResult<Self> {
        let game_config = Arc::new(GameConfig {
            scherbius_starting: config.scherbius_starting,
            scherbius_deal: config.scherbius_deal,
            turing_starting: config.turing_starting,
            turing_deal: config.turing_deal,
            victory_points: config.victory_points,
            n_battles: config.n_battles,
            encryption_cost: config.encryption_cost,
            encryption_code_len: config.encryption_code_len,
            encryption_vocab_size: config.encryption_vocab_size,
            verbose: config.verbose,
            max_vp: config.max_vp,
            max_draw: config.max_draw,
        });

        let game_state = GameState::new(game_config);
        Ok(PyGameState { inner: game_state })
    }

    pub fn step(&mut self, 
        turing_strategy: Vec<Cards>, 
        scherbius_strategy: Vec<Cards>,
        turing_guesses: Vec<Vec<u32>>, 
        reencrypt: bool) -> PyResult<()> {

        self.inner.step(&ScherbiusAction { strategy: scherbius_strategy, encryption: reencrypt },
                        &TuringAction { strategy: turing_strategy, guesses: turing_guesses });

        Ok(())
    }

    pub fn is_won(&self) -> bool {
        self.inner.winner != Actor::Null
    }

    // Expose game state details to Python
    pub fn turing_points(&self) -> u32 {
        self.inner.turing_points
    }

    pub fn scherbius_points(&self) -> u32 {
        self.inner.scherbius_points
    }

    pub fn encryption_broken(&self) -> bool {
        self.inner.encryption_broken
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
        // need to convert rewards to a format that can be used in Python
        let rewards = self.inner.rewards.clone();
        
        let mut victory_points: Vec<u32> = Vec::new();
        let mut new_cards: Vec<Vec<u32>> = Vec::new();

        // we need to keep the info about which battle the reward is for
        for reward in rewards {
            match reward {
                Reward::VictoryPoints(v) => {
                    victory_points.push(v);
                    new_cards.push(vec![]);
                },
                Reward::NewCards(cards) => {
                    new_cards.push(cards);
                    victory_points.push(0);
                },
                _ => (),
            }
        }           

        (new_cards, victory_points)

    }

    pub fn turing_observation(&mut self, scherbius_strategy: Vec<Vec<u32>>) -> (Vec<u32>, Vec<Vec<u32>>) {
        // mut self as we need to update the state of the enigma machine
        let intercepted_scherbius_strategy = self.inner.intercept_scherbius_strategy(&scherbius_strategy);
        (self.inner.turing_hand.clone(), intercepted_scherbius_strategy)
    }

    pub fn scherbius_observation(&self) -> Vec<u32> {
        self.inner.scherbius_hand.clone()
    }
}

#[pymodule]
fn turing_vs_scherbius(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<PyGameConfig>()?;

    Ok(())
}