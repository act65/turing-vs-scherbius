use rand::Rng;
use std::fmt;
// use std::cmp::Ordering;

#[derive(Debug)]
pub struct GameConfig {
    scherbius_starting: u32,
    scherbius_deal: u32,
    scherbius_hand_limit: u32,

    turing_starting: u32,
    turing_deal: u32,
    turing_hand_limit: u32,

    victory_points: u32,
    n_battles: u32,
}

// default game config
impl GameConfig {
    pub fn new() -> GameConfig {
        GameConfig {
            scherbius_starting: 6,
            scherbius_deal: 3,
            scherbius_hand_limit: 4,
            turing_starting: 4,
            turing_deal: 2,
            turing_hand_limit: 4,
            victory_points: 15,
            n_battles: 3,
        }
    }
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
            // deal inital hands
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
        turing_action: &Action,
        scherbius_action: &Action,
        rewards: &Rewards) {

    // each player gets some new cards
    let new_cards = draw_cards(game_config.scherbius_deal);
    self.scherbius_hand.extend_from_slice(&new_cards);

    let new_cards = draw_cards(game_config.turing_deal);
    self.turing_hand.extend_from_slice(&new_cards);

    // let results = resolve_battles(game_state, scherbius_action, turing_action);
    
    // distribute the rewards
    for i in 0..game_config.n_battles {
        let (reward_type, value) = rewards.i;
        let winner = results[i];

        if winner == 0 {
            if reward_type == 0 {
                let new_cards = draw_cards(value);
                self.scherbius_hand.extend_from_slice(&new_cards);
                }
            else {
                self.scherbius_points = self.scherbius_points + value;
            }
        }
        else {
            if reward_type == 0 {
                let new_cards = draw_cards(value);
                self.turing_hand.extend_from_slice(&new_cards);
                }
            else {
                self.turing_points = self.turing_points + value;
            }
        }
    }

    }
}

#[derive(Debug)]
pub struct Rewards {
    pub battle1: (bool, u32),
    pub battle2: (bool, u32),
    pub battle3: (bool, u32),
}

impl fmt::Display for Rewards {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "B1: {:?}\nB2: {:?}\nB3: {:?}", self.battle1, self.battle2, self.battle3)
    }
}

#[derive(Debug)]
pub struct Action {
    pub chosen_cards: u32,
}

fn random_rewards(n: u32)->Rewards {
    let mut rewards = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..n {
        let reward_type = rng.gen::<bool>();
        let reward_value: u32 = rng.gen_range(0..4);
        rewards.push((reward_type, reward_value))
    }
    Rewards {
        battle1: rewards[0],
        battle2: rewards[1],
        battle3: rewards[2],
    }
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

pub type Player = fn(&GameState) -> Action;

pub fn play(
        game_config: GameConfig,
        mut game_state: GameState, 
        sherbius: Player, 
        turing: Player) {

    let mut counter: u32 = 0;

    loop {
        // what is being played for this round?
        let rewards = random_rewards(game_config.n_battles);
        println!("{}", rewards);
        println!("{:?}", game_state);

        counter = counter + 1;
        println!("Counter: {}", counter);

        // Sherbius plays first
        let scherbius_action = sherbius(&game_state);
        println!("{:?}", scherbius_action);

        // Turing plays second
        let turing_action = turing(&game_state);
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

// fn battle_result() {
//     match turing_value.cmp(&scherbius_value) {
//         Ordering::Less => 1,
//         Ordering::Greater => 0,
//         Ordering::Equal => -1,
//     };
// }