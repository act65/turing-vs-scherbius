use std::io;
use rand::{thread_rng, Rng};

use turing_vs_scherbius::{GameState, Action, Reward};


fn get_user_input()->u32 {
    let mut choice: String = String::new();
    io::stdin()
        .read_line(&mut choice)
        .expect("Failed to read line");

    let choice: u32 = choice.trim().parse().expect("Please type a number!");
    choice
}

pub fn user_input_player(
        observation: &GameState, 
        rewards: &Vec<Reward>)
        ->Vec<Action> {
    println!("{:?}", observation);
    println!("{:?}", rewards);

    let mut actions: Vec<Action> = Vec::new();

    for _ in 1..rewards.len() {
        let choice: u32 = get_user_input();
        println!("{:?}", choice);
        let action = Action {
            card1: choice
        };
        actions.push(action);
    }

    actions
}

pub fn random_player(
        observation: &GameState, 
        rewards: &Vec<Reward>)
        ->Vec<Action>{

    let mut rng = thread_rng();
    let mut actions: Vec<Action> = Vec::new();

    for _ in 1..rewards.len() {
        let choice: u32 = rng.gen_range(0..10);
        let action = Action {
            card1: choice
        };
        actions.push(action);
    }

    actions
}