use std::io;
use turing_vs_scherbius::{GameState, Action};

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
