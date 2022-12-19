use std::fmt;
use std::cmp::Ordering;
use std::iter::zip;
use rand::{
    thread_rng,
    Rng,
    rngs::ThreadRng
};
use rand::prelude::SliceRandom;
use pyo3::prelude::*;

pub mod enigma;
use serde::{Serialize, Deserialize};

use std::fs;
use serde_json::from_str;

#[pyclass]
#[derive(Serialize, Deserialize, Debug)]
pub struct GameConfig {
    #[pyo3(get, set)]
    pub scherbius_starting: u32,
    #[pyo3(get, set)]
    pub scherbius_deal: u32,
    #[pyo3(get, set)]
    pub turing_starting: u32,
    #[pyo3(get, set)]
    pub turing_deal: u32,
    #[pyo3(get, set)]
    pub victory_points: u32, // how many victory points required to win
    #[pyo3(get, set)]
    pub n_battles: u32,
    #[pyo3(get, set)]
    pub encryption_cost: u32, // re-encrypting costs victory points
    #[pyo3(get, set)]
    pub encryption_code_len: u32,
    #[pyo3(get, set)]
    pub max_vp: u32,
    #[pyo3(get, set)]
    pub max_draw: u32,
    #[pyo3(get, set)]
    pub verbose: bool,
}

#[pyfunction]
pub fn read_config(path: String) -> GameConfig {
    let data = fs::read_to_string(path).expect("Unable to read file");
    let game_config: GameConfig = serde_json::from_str(&data).expect("JSON was not well-formatted");
    return game_config
}

#[pymethods]
impl GameConfig {
    #[new]
    pub fn new(scherbius_starting: u32,
        scherbius_deal: u32,
        turing_starting: u32,
        turing_deal: u32,
        victory_points: u32,
        n_battles: u32,
        encryption_cost: u32,
        encryption_code_len: u32,
        max_vp: u32,
        max_draw: u32,
        verbose: bool) -> GameConfig {
        GameConfig {
            scherbius_starting: scherbius_starting,
            scherbius_deal: scherbius_deal,
            turing_starting: turing_starting,
            turing_deal: turing_deal,
            victory_points: victory_points,
            n_battles: n_battles,
            encryption_cost: encryption_cost,
            encryption_code_len: encryption_code_len,
            max_vp: max_vp,
            max_draw: max_draw,
            verbose: verbose
        }
    }
}




// #[pyclass]
#[derive(Debug)]
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

}

// impl fmt::Display for GameState {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "Turing: {:?}\nSherbius: {:?}", self.turing_hand, self.scherbius_hand)
//     }
// }

// #[pymethods]
impl GameState{
    // #[new]
    fn new(game_config: &GameConfig) -> GameState {
        let mut rng = rand::thread_rng();
        let a: u32 = rng.gen_range(1..10);
        let b: u32 = rng.gen_range(1..10);
        
        GameState{
            // deal initial random hands
            scherbius_hand: draw_cards(game_config.scherbius_starting),
            turing_hand: draw_cards(game_config.turing_starting),
        
            encryption_broken: false,
            encryption: vec![a, b],

            turing_points: 0,
            scherbius_points: 0,

            encoder: enigma::EasyEnigma::new(10),
            winner: Actor::Null,

        }
    }

    fn intercept_scherbius_strategy(&mut self, strategy: &Vec<Cards>) -> Vec<Cards> {

        if self.encryption_broken {
            return strategy.clone()}
        else {
            let mut encrypted_strategy: Vec<Cards> = Vec::new();
            for h in strategy {
                encrypted_strategy.push(self.encoder.call(&h));
            }
            return encrypted_strategy}
    }

    fn step(
        &mut self,
        game_config: &GameConfig, 
        scherbius_action: &ScherbiusAction,
        turing_action: &TuringAction,
        rewards: &Vec<Reward>) {

    // each player gets some new cards
    let new_cards = draw_cards(game_config.scherbius_deal);
    self.scherbius_hand.extend_from_slice(&new_cards);

    let new_cards = draw_cards(game_config.turing_deal);
    self.turing_hand.extend_from_slice(&new_cards);

    // remove cards played from hands
    // TODO move into fn
    for c in scherbius_action.strategy.iter() {
        for i in c.iter() {
            let index = self.scherbius_hand.iter().position(|y| *y == *i).unwrap();
            self.scherbius_hand.remove(index);    
        }
    }
    for c in turing_action.strategy.iter() {
        for i in c.iter() {
            let index = self.turing_hand.iter().position(|y| *y == *i).unwrap();
            self.turing_hand.remove(index);    
        }
    }
    // remove guesses from turing's hand
    for g in turing_action.guesses.iter() {
        for i in g.iter() {
            let index = self.turing_hand.iter().position(|y| *y == *i).unwrap();
            self.turing_hand.remove(index);        
        }
    }

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
    // - no cards vs cards means cards wins

    // distribute the rewards
    for (result, reward) in zip(results, rewards) {
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
    if self.scherbius_points >= game_config.encryption_cost && scherbius_action.encryption 
        {self.encoder.set([rng.gen_range(0..10), rng.gen_range(0..10)]);
        self.encoder.reset();
        self.encryption_broken=false};

    // check if a player has won
    if self.scherbius_points >= game_config.victory_points {
        self.winner = Actor::Scherbius;}
    else if self.turing_points >= game_config.victory_points {
        self.winner = Actor::Turing;
    }

    }
}

#[pymodule]
fn turing_vs_scherbius(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(read_config, m)?)?;

    m.add_class::<GameConfig>()?;
    // m.add_class::<GameState>()?;

    Ok(())
}


pub type ScherbiusPlayer = fn(&Vec<u32>, &Vec<Reward>) -> ScherbiusAction;
pub type TuringPlayer = fn(&Vec<u32>, &Vec<Reward>, &Vec<Cards>) -> TuringAction;

pub type Cards = Vec<u32>;
pub type EncryptionCode = Vec<u32>;

#[derive(Debug, PartialEq)]
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

#[derive(Debug)]
pub enum Reward {
    VictoryPoints(u32),
    NewCards(Vec<u32>),
    Null,
}

fn sample_battle_reward(max_vp: u32, max_draw: u32, rng: &mut ThreadRng) -> Reward {
    match rng.gen_range(0..2) {
        // TODO distribution should make larger values less likely
        // TODO want a parameter to control the max value in GameConfig?!
        // TODO move rng to be an arg?
        0 => Reward::VictoryPoints(rng.gen_range(1..max_vp)),
        1 => Reward::NewCards(draw_cards(rng.gen_range(1..max_draw))),
        _ => Reward::Null,
    }
}

fn random_rewards(n: u32, max_vp: u32, max_draw: u32, rng: &mut ThreadRng)->Vec<Reward> {
    let mut rewards: Vec<Reward> = Vec::new();

    for _ in 0..n {
        let reward: Reward = sample_battle_reward(max_vp, max_draw, rng);
        rewards.push(reward)
    }
    rewards
}

#[derive(Debug)]
pub struct TuringAction {
    pub strategy: Vec<Cards>,
    pub guesses: Vec<EncryptionCode>,
}

#[derive(Debug)]
pub struct ScherbiusAction {
    pub strategy: Vec<Cards>,
    pub encryption: bool,
}

fn draw_cards(n: u32)->Cards {
    let mut cards = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let value: u32 = rng.gen_range(1..11);
        cards.push(value)
    }
    cards
}

pub fn play(
    game_config: GameConfig,
    sherbius: ScherbiusPlayer, 
    turing: TuringPlayer) -> Actor{

    let mut game_state = GameState::new(&game_config);

    // TODO move all fns to use this rng?
    let mut rng = thread_rng();

    loop {
        // what is being played for this round?
        let rewards = random_rewards(
            game_config.max_vp, 
            game_config.max_draw,
            game_config.n_battles, &mut rng);

        // Sherbius plays first
        let scherbius_action = sherbius(&game_state.scherbius_hand, &rewards);
        let intercepted_scherbius_strategy = game_state.intercept_scherbius_strategy(&scherbius_action.strategy);
        // Turing plays second
        let turing_action = turing(&game_state.turing_hand, &rewards, &intercepted_scherbius_strategy);

        // check_action_validity(turing_action);
        // check_action_validity(sherbius_action);

        game_state.step(
                &game_config,
                &scherbius_action, 
                &turing_action,
                &rewards);

        if game_config.verbose {
            println!("{:?}", game_state);
        }

        if game_state.winner != Actor::Null {
            break
        }
        
    }
    if game_config.verbose {
        println!("Winner: {:?}", game_state.winner);
    }
    game_state.winner
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step() {
        {}
    }
}