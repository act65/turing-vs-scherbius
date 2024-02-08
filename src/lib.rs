use std::fmt;
use std::cmp::Ordering;
use std::iter::zip;
use rand::{
    thread_rng,
    Rng,
    rngs::ThreadRng
};

pub mod enigma;
pub mod utils;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
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

    rng: ThreadRng,

}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Turing: {:?}\nSherbius: {:?}", self.turing_hand, self.scherbius_hand)
    }
}

impl GameState{
    fn new(game_config: &GameConfig) -> GameState {
        let mut rng = rand::thread_rng();

        
        GameState{
            // deal initial random hands
            scherbius_hand: draw_cards(game_config.scherbius_starting, &mut rng),
            turing_hand: draw_cards(game_config.turing_starting, &mut rng),
        
            encryption_broken: false,
            encryption: utils::sample_random_ints(game_config.encryption_code_len, game_config.encryption_vocab_size, &mut rng),

            turing_points: 0,
            scherbius_points: 0,

            encoder: enigma::EasyEnigma::new(10, &mut rng),
            winner: Actor::Null,

            rng: rng,

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
        game_config: &GameConfig, 
        scherbius_action: &ScherbiusAction,
        turing_action: &TuringAction,
        rewards: &Vec<Reward>) {

    // each player gets some new cards
    let new_cards = draw_cards(game_config.scherbius_deal, &mut self.rng);
    self.scherbius_hand.extend_from_slice(&new_cards);

    let new_cards = draw_cards(game_config.turing_deal, &mut self.rng);
    self.turing_hand.extend_from_slice(&new_cards);

    // remove cards played from hands
    utils::remove_played_cards_from_hand(&mut self.scherbius_hand, &scherbius_action.strategy);
    utils::remove_played_cards_from_hand(&mut self.turing_hand, &turing_action.strategy);
    utils::remove_played_cards_from_hand(&mut self.turing_hand, &turing_action.guesses);

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
        1 => Reward::NewCards(draw_cards(rng.gen_range(1..max_draw), rng)),
        _ => Reward::Null,
    }
}

fn random_rewards(n: u32, max_vp: u32, max_draw: u32, rng: &mut ThreadRng)->Vec<Reward> {
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

fn draw_cards(n: u32, rng: &mut ThreadRng) -> Cards {
    let mut cards = Vec::new();

    for _ in 0..n {
        let value: u32 = rng.gen_range(1..11);
        cards.push(value)
    }
    cards
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

pub fn play(
    game_config: GameConfig,
    sherbius: ScherbiusPlayer, 
    turing: TuringPlayer) -> Actor{

    let mut game_state = GameState::new(&game_config);

    loop {
        // what is being played for this round?
        let rewards = random_rewards(
            game_config.n_battles,
            game_config.max_vp, 
            game_config.max_draw,
             &mut game_state.rng);

        // Sherbius plays first
        let scherbius_action = sherbius(&game_state.scherbius_hand, &rewards);
        let intercepted_scherbius_strategy = game_state.intercept_scherbius_strategy(&scherbius_action.strategy);
        // Turing plays second
        let turing_action = turing(&game_state.turing_hand, &rewards, &intercepted_scherbius_strategy);

        check_action_validity(&Action::ScherbiusAction(scherbius_action.clone()), &game_state.scherbius_hand);
        check_action_validity(&Action::TuringAction(turing_action.clone()), &game_state.turing_hand);

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