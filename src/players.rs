use std::io;
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;

use turing_vs_scherbius::{GameState, ScherbiusAction, TuringAction, Reward, Actor, Cards};


// fn get_user_input()->u32 {
//     let mut choice: String = String::new();
//     io::stdin()
//         .read_line(&mut choice)
//         .expect("Failed to read line");

//     let choice: u32 = choice.trim().parse().expect("Please type a number!");
//     choice
// }

// pub fn user_input_player(
//         observation: &GameState, 
//         rewards: &Vec<Reward>)
//         ->Vec<Action> {
//     println!("{:?}", observation);
//     println!("{:?}", rewards);

//     let mut actions: Vec<Action> = Vec::new();

//     for i in 0..rewards.len() {
//         println!("Battle {}", i);
//         let choice: u32 = get_user_input();
//         println!("{:?}", choice);
//         let action = Action {
//             card1: choice
//         };
//         actions.push(action);
//     }

//     actions
// }


fn draw(deck: &mut Cards, n: usize)->Cards {
    let drawn: Vec<u32> = Vec::new();
    for i in 0..n {
        let x = deck.choose(&mut rand::thread_rng());
        match x {
            Some(x) => {
                drawn.push(*x);
                deck.remove(*x as usize);}
            None => ()
        }
        }
    drawn
}

fn get_rnd_strategy(hand: &mut Cards, n: usize)->Vec<Cards>{
    let mut strategy: Vec<Cards> = Vec::new();

    for i in 0..n {
        let choice = draw(hand, 1);
        strategy.push(choice);
    }
    strategy
}

pub fn random_scherbius_player(
        observation: &mut GameState, 
        rewards: &Vec<Reward>)
        ->ScherbiusAction{

    let strategy = get_rnd_strategy(&observation.scherbius_hand, rewards.len());

    ScherbiusAction {
        strategy: strategy,
        encryption: None,
    }
}

pub fn random_turing_player(
    observation: &mut GameState, 
    rewards: &Vec<Reward>,
    encrypted_cards: &Vec<Cards>)
    ->TuringAction{

    let strategy = get_rnd_strategy(&observation.turing_hand, rewards.len());

    TuringAction {
        strategy: strategy,
        guesses: None,
    }
}