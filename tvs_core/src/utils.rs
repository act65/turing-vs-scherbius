use rand::{
    Rng,
    rngs::ThreadRng,
    thread_rng
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;

type Cards = Vec<u32>;

use std::io;

pub fn draw_cards(n: u32, rng: &mut StdRng) -> Cards {
    let mut cards = Vec::new();

    for _ in 0..n {
        let value: u32 = rng.gen_range(1..11);
        cards.push(value)
    }
    cards
}


// fns for human players
pub fn get_user_input(prompt: &str) -> Option<u32> {
    // Get user input from the command line
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

pub fn choose_from_set(deck: &[u32], prompt: &str) -> Cards {
    // Choose a subset of cards from a deck
    // Returns a vector of chosen cards
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

pub fn draw_from_set(deck: &mut Cards, n: usize)->Cards {
    // Draw n cards from a deck
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

// for random players
pub fn get_rnd_strategy(hand: &mut Cards, n: usize) -> Vec<Cards> {
    // randomly choose a strategy from the hand
    let mut strategy: Vec<Cards> = Vec::new();
    while !hand.is_empty() {
        let m = hand.len();
        // Calculate max possible size for this subset
        // let max_subset_size = hand.len().max(1); // At least one card subset
        let subset_size = thread_rng().gen_range(1..=m);
        let choice = draw_from_set(hand, subset_size);
        strategy.push(choice);
    }

    // if len of strategy is less than n
    // append empty vectors to strategy
    while strategy.len() < n {
        strategy.push(Vec::new());
    }

    strategy
}

pub fn random_reencryption() -> bool {
    thread_rng().gen_bool(1.0 / 5.0) // 1 in 5 chance of being true
}

// other utils
pub fn sample_random_ints(n: u32, max: u32, rng: &mut StdRng) -> Vec<u32> {
    (0..n).map(|_| rng.gen_range(1..max)).collect()
}

fn flatten<T>(v: &Vec<Vec<T>>) -> Vec<T>
where
    T: Copy, // Ensure T implements Copy, so we can dereference safely
{
    let mut result: Vec<T> = Vec::new();
    for x in v {
        for y in x {
            result.push(*y);
        }
    }
    result
}

pub fn remove_played_cards_from_hand(hand: &mut Cards, played: &Vec<Cards>) {
    let flat_played: Vec<u32> = flatten(played);
    for c in flat_played.iter() {
        if let Some(index) = hand.iter().position(|&y| y == *c) {
            hand.remove(index);
        }
    }
}

pub fn is_subset_of_hand(strategy: &Vec<Cards>, hand: &Cards) -> bool {
    // check if strategy is a unique subset of hand
    // not just a subset
    let flat_strategy: Vec<u32> = flatten(strategy);
    let flat_hand: Vec<u32> = hand.clone();
    is_unique_subset(&flat_strategy, &flat_hand)
}

fn is_unique_subset(a: &Vec<u32>, b: &Vec<u32>) -> bool {
    // check if a is a unique subset of b
    // not just a subset
    let mut b_copy = b.clone();
    for c in a.iter() {
        // if c is in b_copy
        // then remove it from b_copy
        if let Some(index) = b_copy.iter().position(|&y| y == *c) {
            b_copy.remove(index);
        // if c is not in b_copy
        // then a is not a subset of b
        } else {
            return false
        }
    }
    return true
}

// tests

#[cfg(test)]
// test random ints
mod test_random_ints {

    #[test]
    fn test_sample_random_ints() {
        let mut rng = thread_rng();
        let n = 10;
        let max = 100;
        let result = sample_random_ints(n, max, &mut rng);
        assert_eq!(result.len(), n as usize);
        for x in result {
            assert!(x < max);
        }
    }
}
// test flatten
mod test_flatten {

    #[test]
    fn test_flatten() {
        let v = vec![vec![1, 2], vec![3, 4]];
        let result = flatten(&v);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }
}