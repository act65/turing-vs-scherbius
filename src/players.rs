use std::io;
use rand::{thread_rng, Rng};
// use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

use turing_vs_scherbius::{ScherbiusAction, TuringAction, Reward, Cards};


fn get_user_input()->Cards {
    let mut choice: String = String::new();
    io::stdin()
        .read_line(&mut choice)
        .expect("Failed to read line");

    let choice: u32 = choice.trim().parse().expect("Please type a number!");
    vec![choice]
}

pub fn user_input_player(
        scherbius_hand: &Vec<u32>, 
        rewards: &Vec<Reward>)
        ->ScherbiusAction{

    // need to verify that the user's chosen cards are actually in their hand.
    // if not keep asking for a valid choice 
    println!("Scherbius hand. {:?}", scherbius_hand);
    println!("Rewards. {:?}", rewards);

    let mut strategy: Vec<Cards> = Vec::new();

    for i in 0..rewards.len() {
        println!("Battle {}", i);
        let choice: Cards = get_user_input();
        println!("{:?}", choice);
        strategy.push(choice);
    }

    ScherbiusAction {
        strategy: strategy,
        encryption: true,
    }
    // need a callback to print results?
}
pub fn turing_user_player(
    turing_hand: &Vec<u32>, 
    rewards: &Vec<Reward>,
    encrypted_cards: &Vec<Cards>)
    ->TuringAction{

    println!("Rewards");
    println!("{:?}", turing_hand);
    println!("Hand");
    println!("{:?}", rewards);

    let x = get_user_input();
    let y = get_user_input();
    
    let mut hand = turing_hand.clone();
    let guesses = get_rnd_guesses(&mut hand, 1);
    let strategy = get_rnd_strategy(&mut hand, rewards.len());
    
    println!("{:?}", strategy);


    TuringAction {
        strategy: strategy,
        guesses: guesses,
    }
}


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
                deck.remove(index);
                }
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

pub fn random_reencryption()->bool {
    match thread_rng().gen_range(0..5) {
        0 => true,
        _ => false,
    }
}

fn get_rnd_guesses(hand: &mut Cards, n: usize)->Vec<EncryptionCode>{
    let mut guesses: Vec<EncryptionCode> = Vec::new();

    for _ in 0..n {
        let choice = draw(hand, 2);
        guesses.push(vec![choice[0], choice[1]]);
    }
    guesses
}

pub fn random_scherbius_player(
        scherbius_hand: &Vec<u32>, 
        rewards: &Vec<Reward>)
        ->ScherbiusAction{

    let mut hand = scherbius_hand.clone();
    let strategy = get_rnd_strategy(&mut hand, rewards.len());
    let encrypt = random_reencryption();

    ScherbiusAction {
        strategy: strategy,
        encryption: encrypt,
    }
}

pub fn random_turing_player(
    turing_hand: &Vec<u32>, 
    rewards: &Vec<Reward>,
    encrypted_cards: &Vec<Cards>)
    ->TuringAction{

    let mut hand = turing_hand.clone();
    let guesses = get_rnd_guesses(&mut hand, 1);
    let strategy = get_rnd_strategy(&mut hand, rewards.len());

    TuringAction {
        strategy: strategy,
        guesses: guesses,
    }
}