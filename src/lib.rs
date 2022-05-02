use std::fmt;
use std::cmp::Ordering;
use std::iter::zip;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

#[derive(Debug)]
pub struct GameConfig {
    pub scherbius_starting: u32,
    pub scherbius_deal: u32,
    pub scherbius_hand_limit: u32,

    pub turing_starting: u32,
    pub turing_deal: u32,
    pub turing_hand_limit: u32,

    pub victory_points: u32,
    pub n_battles: u32,
}

#[derive(Debug)]
pub struct GameState {
    pub turing_hand: Vec<u32>,
    pub scherbius_hand: Vec<u32>,

    pub encryption: u32,

    pub turing_points: u32,
    pub scherbius_points: u32,
}

// initial game state
impl GameState {
    pub fn new(game_config: &GameConfig) -> GameState {
        GameState{
            // deal inital random hands
            scherbius_hand: draw_cards(game_config.scherbius_starting),
            turing_hand: draw_cards(game_config.turing_starting),
        
            encryption: 2,

            turing_points: 0,
            scherbius_points: 0,
        }
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Turing: {:?}\nSherbius: {:?}", self.turing_hand, self.scherbius_hand)
    }
}

impl GameState{
    pub fn step(
        &mut self,
        game_config: &GameConfig, 
        turing_actions: &Vec<Action>,
        scherbius_actions: &Vec<Action>,
        rewards: &Vec<Reward>) {

    // each player gets some new cards
    let new_cards = draw_cards(game_config.scherbius_deal);
    self.scherbius_hand.extend_from_slice(&new_cards);

    let new_cards = draw_cards(game_config.turing_deal);
    self.turing_hand.extend_from_slice(&new_cards);

    // resolve battles
    let results: Vec<_> = zip(scherbius_actions.iter(), turing_actions.iter())
        .map(|(a1, a2)|battle_result(a1, a2))
        .filter(|x| x.is_some())
        .map(|x| x.unwrap())
        .collect();

    // distribute the rewards
    for (result, reward) in zip(results, rewards) {
        match result {
            Actor::Turing => 
                {match reward {
                    Reward::VictoryPoints(v) => self.turing_points = self.turing_points + v,
                    Reward::NewCards(cards) => self.turing_hand.extend_from_slice(&cards)
                    }},
            Actor::Scherbius => 
                {match reward {
                    Reward::VictoryPoints(v) => self.scherbius_points = self.scherbius_points + v,
                    Reward::NewCards(cards) => self.scherbius_hand.extend_from_slice(&cards)
                }}
            _ => ()
        }

    }

    }
}

pub type Player = fn(&GameState, &Vec<Reward>) -> Vec<Action>;

#[derive(Debug)]
enum Actor {
    Scherbius,
    Turing,
    // Scherbius(Player),
    // Turing(Player),
}


fn battle_result(
    sherbius_action: &Action, 
    scherbius_action: &Action) -> Option<Actor> {
    match sherbius_action.card1.cmp(&scherbius_action.card1) {
        Ordering::Less => Some(Actor::Turing),
        Ordering::Greater => Some(Actor::Scherbius),
        Ordering::Equal => None,
    }
}

#[derive(Debug)]
pub enum Reward {
    VictoryPoints(u32),
    NewCards(Vec<u32>),
}

impl Distribution<Reward> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Reward {
        match rng.gen_range(0..2) {
            0 => Reward::VictoryPoints(rng.gen_range(0..4)),
            1 => Reward::NewCards(draw_cards(rng.gen_range(0..4))),
            _ => Reward::VictoryPoints(100)
        }
    }
}

// impl fmt::Display for Rewards {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "B1: {:?}\nB2: {:?}\nB3: {:?}", self.battle1, self.battle2, self.battle3)
//     }
// }

#[derive(Debug)]
pub struct Action {
    pub card1: u32,
    // pub card2: u32,
}

fn random_rewards(n: u32)->Vec<Reward> {
    let mut rewards = Vec::new();

    for _ in 0..n {
        let reward: Reward = rand::random();

        rewards.push(reward)
    }
    rewards
}

fn draw_cards(n: u32)->Vec<u32> {
    let mut cards = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..n {
        let value: u32 = rng.gen_range(0..11);
        cards.push(value)
    }
    cards
}

pub fn play(
        game_config: GameConfig,
        mut game_state: GameState, 
        sherbius: Player, 
        turing: Player) {

    loop {
        // what is being played for this round?
        let rewards = random_rewards(game_config.n_battles);

        // Sherbius plays first
        let scherbius_action = sherbius(&game_state, &rewards);

        // Turing plays second
        let turing_action = turing(&game_state, &rewards);
        // gamestate = gamestate.update(&action);

        // check_action_validity(turing_action);
        // check_action_validity(sherbius_action);

        game_state.step(
                &game_config,
                &scherbius_action, 
                &turing_action, 
                &rewards);

        // check if a player has won
        if game_state.scherbius_points > 2 {
            let winner: u32 = 0;
            break;}
        else if game_state.turing_points > 2 {
            let winner: u32 = 1;
            break;
        }
    }
}