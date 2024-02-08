use std::io;
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;

use turing_vs_scherbius::{
    ScherbiusAction, 
    TuringAction, 
    Reward, 
    Cards, 
    EncryptionCode
};

fn get_user_input(prompt: &str) -> Option<u32> {
    println!("{}", prompt);
    let mut choice = String::new();
    io::stdin()
        .read_line(&mut choice)
        .expect("Failed to read line");

    let choice = choice.trim();
    if choice.is_empty() {
        None
    } else {
        choice.parse().ok()
    }
}

pub fn turing_human_player(
    turing_hand: &Vec<u32>, 
    rewards: &Vec<Reward>,
    encrypted_cards: &Vec<Cards>)
    ->TuringAction{

    println!("Rewards");
    println!("{:?}", rewards);
    println!("Hand");
    println!("{:?}", turing_hand);
    println!("Scherbius strategy");
    println!("{:?}", encrypted_cards);

    let guess_prompt: String = "Choose your guess (enter to skip)".to_string();
    let guesses = choose_from_set(&turing_hand, &guess_prompt);

    let strategy_prompt: String = "Choose your strategy (enter to end)".to_string();
    
    let strategy = get_strategy_from_hand(&turing_hand, &rewards, &strategy_prompt); // Reusing function

    TuringAction {
        strategy: strategy,
        guesses: vec![guesses],
    }
}

pub fn scherbius_human_player(
    scherbius_hand: &Vec<u32>, 
    rewards: &Vec<Reward>)
    ->ScherbiusAction{

    println!("Rewards");
    println!("{:?}", rewards);
    println!("Hand");
    println!("{:?}", scherbius_hand);

    let mut hand = vec![0, 1];
    let encryption_prompt: String = "Choose whether to re-encrypt (0->False, 1->True)".to_string();
    let choice = get_user_input(&encryption_prompt);
    let encryption = match choice.into_iter().nth(0) {
        Some(1) => true,
        _ => false,
    };

    let strategy_prompt: String = "Choose your strategy (enter to end)".to_string();
    
    let strategy = get_strategy_from_hand(&scherbius_hand, &rewards, &strategy_prompt); // Reusing function
    
    ScherbiusAction {
        strategy: strategy,
        encryption: encryption,
    }
}

fn get_strategy_from_hand(hand: &[u32], rewards: &[Reward], prompt: &str) -> Vec<Cards> {
    rewards.iter().map(|r| {
        println!("Rewards: {:?}", r);
        println!("Hand {:?}", hand);
        choose_from_set(hand, prompt)  // Fixed to pass a slice reference
    }).collect()
}

fn choose_from_set(deck: &[u32], prompt: &str) -> Cards {
    let mut chosen: Vec<u32> = Vec::new();
    loop {
        if let Some(x) = get_user_input(prompt) {
            if deck.contains(&x) {
                chosen.push(x);
            } else {
                println!("Please choose a number from {:?}", deck);
            }
        } else {
            break; // Exit the loop if no input was given (user pressed enter)
        }
    }
    chosen
}

pub fn remove_chosen_items_from_hand(hand: &mut Vec<u32>, chosen: &[u32]) {
    for &item in chosen {
        if let Some(index) = hand.iter().position(|&x| x == item) {
            hand.remove(index);
        }
    }
}

fn draw_from_set(deck: &mut Cards, n: usize)->Cards {
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

fn get_rnd_strategy(hand: &mut Cards, n: usize) -> Vec<Cards> {
    let mut strategy: Vec<Cards> = Vec::new();
    while !hand.is_empty() {
        // Calculate max possible size for this subset
        let max_subset_size = hand.len().max(1); // At least one card subset
        let subset_size = thread_rng().gen_range(1..=max_subset_size);
        let choice = draw_from_set(hand, subset_size);
        strategy.push(choice);
    }
    strategy
}

pub fn random_reencryption() -> bool {
    thread_rng().gen_bool(1.0 / 5.0) // 1 in 5 chance of being true
}

fn get_rnd_guesses(hand: &mut Cards, n: usize)->Vec<EncryptionCode>{
    let mut guesses: Vec<EncryptionCode> = Vec::new();

    for _ in 0..n {
        // problem? how many cards in each guess?
        let choice = draw_from_set(hand, 2);
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