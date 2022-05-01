use rand::Rng;
use std::io;

#[derive(Debug)]
pub struct GameState {
    pub turing_victory: bool,
    pub scherbuis_victory: bool,

    pub turing_hand: Vec<u32>,
    pub sherbius_hand: Vec<u32>,

    pub encryption: u32,

}

impl GameState {
    pub fn new() -> GameState {
        GameState{
            turing_victory: true,
            scherbuis_victory: true,
        
            turing_hand: vec![1, 2, 3],
            sherbius_hand: vec![1, 2, 3],
        
            encryption: 2,
        }
    }
}

#[derive(Debug)]
struct Rewards {
    battle1: (bool, u32),
    battle2: (bool, u32),
    battle3: (bool, u32),
}

#[derive(Debug)]
pub struct Action {
    chosen_cards: u32,
}

fn random_rewards()->Rewards {
    let mut rewards = Vec::new();
    let mut rng = rand::thread_rng();

    for n in 0..3 {

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

pub fn user_input_player(observation: &GameState)->Action {
    println!("{:?}", observation);
    let mut choice: String = String::new();

    io::stdin()
    .read_line(&mut choice)
    .expect("Failed to read line");

    let choice: u32 = choice.trim().parse().expect("Please type a number!");

    Action {
        chosen_cards: choice
    }
}

pub type Player = fn(&GameState) -> Action;

pub fn play(
        gamestate: GameState, 
        sherbius: Player, 
        turing: Player)->String {
    loop {
        // what is being played for this round?
        let rewards = random_rewards();

        // Sherbius plays first
        // action = sherbius(&gamestate.observation("Sherbius"));
        let action = sherbius(&gamestate);
        // gamestate = gamestate.update(&action);
        println!("{:?}", action);

        // Turing plays second
        // action = turing(&gamestate.observation("Turing"));
        // gamestate = gamestate.update(&action);
    }
}

