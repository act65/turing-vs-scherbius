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

fn get_user_input(prompt: &String)->Option<u32> {
    println!("{:?}", prompt);
    let mut choice: String = String::new();
    io::stdin()
        .read_line(&mut choice)
        .expect("Failed to read line");

    //  "\n" are automatically added
    if choice.len() == 1 {
        return None
    } else {
        let choice: u32 = choice.trim().parse().expect("Please type a number");
        return Some(choice)    
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

    let mut hand = turing_hand.clone();

    let guess_prompt: String = "Choose your guess".to_string();
    let guesses = choose_from_set(&mut hand, &guess_prompt);

    let strategy_prompt: String = "Choose your strategy".to_string();
    
    let mut strategy: Vec<Cards> = Vec::new();
    for r in rewards {
        println!("Rewards: {:?}", r);
        println!("Hand {:?}", hand);
        strategy.push(choose_from_set(&mut hand, &strategy_prompt));
    }
    
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

    let mut hand = scherbius_hand.clone();
    let strategy_prompt: String = "Choose your strategy".to_string();
    
    let mut strategy: Vec<Cards> = Vec::new();
    for r in rewards {
        println!("Rewards: {:?}", r);
        println!("Hand {:?}", hand);
        strategy.push(choose_from_set(&mut hand, &strategy_prompt));
    }
    
    ScherbiusAction {
        strategy: strategy,
        encryption: encryption,
    }
}

fn choose_from_set(deck: &mut Cards, prompt: &String)->Cards {
    let mut chosen: Vec<u32> = Vec::new();
    while let Some(x) = get_user_input(prompt)  {

        if deck.contains(&x) {
            chosen.push(x);

            let index = deck.iter().position(|x| x == x).unwrap();
            deck.remove(index);
        } else {
            println!("Please choose a number from {:?}", deck);
        }
    }
    return chosen
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

fn get_rnd_strategy(hand: &mut Cards, n: usize)->Vec<Cards>{
    let mut strategy: Vec<Cards> = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..n {
        // pick a random num of cards
        let k: u32 = rng.gen_range(0..2);
        let choice = draw_from_set(hand, k as usize);
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