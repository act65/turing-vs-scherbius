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
    let mut drawn: Vec<u32> = Vec::new();
    // TODO shouldnt create new thread_rng. should pass in as mut arg
    let mut rng = thread_rng();
    for _ in 0..n {
        let x = deck.choose(&mut rng);
        match x {
            Some(x) => {
                drawn.push(*x);
                let index = deck.iter().position(|y| *y == *x).unwrap();
                deck.remove(index);}
            None => ()
        }
        }
    drawn
}

fn get_rnd_strategy(hand: &mut Cards, n: usize)->Vec<Cards>{
    let mut strategy: Vec<Cards> = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..n {
        // pick a random num of cards
        let k: u32 = rng.gen_range(0..2);
        let choice = draw(hand, k as usize);
        strategy.push(choice);
    }
    strategy
}

pub fn random_scherbius_player(
        gamestate: &mut GameState, 
        rewards: &Vec<Reward>)
        ->ScherbiusAction{

    let mut hand = gamestate.scherbius_hand.clone();
    let strategy = get_rnd_strategy(&mut hand, rewards.len());
    gamestate.update(hand, Actor::Scherbius);

    ScherbiusAction {
        strategy: strategy,
        encryption: None,
    }
}

pub fn random_turing_player(
    gamestate: &mut GameState, 
    rewards: &Vec<Reward>,
    encrypted_cards: &Vec<Cards>)
    ->TuringAction{

    let mut hand = gamestate.turing_hand.clone();
    let strategy = get_rnd_strategy(&mut hand, rewards.len());
    gamestate.update(hand, Actor::Turing);

    TuringAction {
        strategy: strategy,
        guesses: None,
    }
}