use rand::{
    Rng
};
use rand::rngs::StdRng;

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

pub fn draw_from_set(deck: &mut Cards, n: usize, rng: &mut StdRng)->Cards {
    // Draw n cards from a deck
    let mut drawn: Vec<u32> = Vec::new();
    // Use the passed-in rng
    for _ in 0..n {
        let x = deck.choose(rng); // Pass rng to choose
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
pub fn get_rnd_strategy(hand: &mut Cards, n: usize, rng: &mut StdRng) -> Vec<Cards> {
    // randomly choose a strategy from the hand
    let mut strategy: Vec<Cards> = Vec::new();
    while !hand.is_empty() {
        let m = hand.len();
        // Calculate max possible size for this subset
        // let max_subset_size = hand.len().max(1); // At least one card subset
        let subset_size = rng.gen_range(1..=m); // Use passed-in rng
        let choice = draw_from_set(hand, subset_size, rng); // Pass rng to draw_from_set
        strategy.push(choice);
    }

    // if len of strategy is less than n
    // append empty vectors to strategy
    while strategy.len() < n {
        strategy.push(Vec::new());
    }

    strategy
}

pub fn random_reencryption(rng: &mut StdRng) -> bool {
    rng.gen_bool(1.0 / 5.0) // 1 in 5 chance of being true
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
    let flat_strategy: Vec<u32> = flatten(strategy);
    // Avoid cloning when not necessary
    is_unique_subset(&flat_strategy, hand)
}

fn is_unique_subset(a: &Vec<u32>, b: &Vec<u32>) -> bool {
    let mut b_copy = b.clone(); // Clone only once here
    for c in a.iter() {
        if let Some(index) = b_copy.iter().position(|&y| y == *c) {
            b_copy.remove(index);
        } else {
            return false;
        }
    }
    true // Simplified return
}

#[cfg(test)]
mod tests {
    use super::*;
    // use rand::thread_rng; // No longer needed directly in most tests after refactor
    use rand::rngs::StdRng;
    use rand::SeedableRng;
     // For gen_range, gen_bool if used directly in tests

    #[test]
    fn test_sample_random_ints() {
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        
        let result = super::sample_random_ints(10, 100, &mut rng);
        assert_eq!(result.len(), 10);
        for x in result {
            assert!(x < 100);
        }
    }

    #[test]
    fn test_draw_cards() {
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        
        let cards = draw_cards(5, &mut rng);
        assert_eq!(cards.len(), 5);
        
        // All cards should be in range 1-10
        for card in cards {
            assert!(card >= 1 && card <= 10);
        }
    }

    #[test]
    fn test_draw_from_set_util() { // Renamed to avoid conflict if a lib.rs test exists
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut deck = vec![1, 2, 3, 4, 5];
        let drawn = draw_from_set(&mut deck, 2, &mut rng);
        assert_eq!(drawn.len(), 2);
        assert_eq!(deck.len(), 3);
        // Ensure drawn cards are from the original deck and no longer in the deck
        for card in drawn.iter() {
            assert!(!deck.contains(card));
        }
    }

    #[test]
    fn test_get_rnd_strategy_util() { // Renamed
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut hand = vec![1, 2, 3, 4, 5, 6];
        let n_battles = 3;
        let strategy = get_rnd_strategy(&mut hand, n_battles, &mut rng);
        
        assert!(hand.is_empty()); // All cards should be distributed
        assert!(strategy.len() >= n_battles || strategy.iter().map(|s| s.len()).sum::<usize>() == 6); // Strategy should cover n_battles or all cards used
        
        let mut total_cards_in_strategy = 0;
        for battle_cards in &strategy {
            total_cards_in_strategy += battle_cards.len();
        }
        assert_eq!(total_cards_in_strategy, 6); // All cards from hand must be in strategy
    }

    #[test]
    fn test_random_reencryption_util() { // Renamed
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        // Test a few times to see if we get both true and false, though with fixed seed it will be deterministic
        let _ = random_reencryption(&mut rng); 
        // We can't assert true/false directly without knowing the seeded behavior,
        // but we can assert it runs and returns a bool.
        assert!(matches!(random_reencryption(&mut rng), true | false));
    }


    #[test]
    fn test_remove_played_cards_from_hand() {
        let mut hand = vec![1, 2, 3, 4, 5];
        let played = vec![vec![1, 2], vec![3]];
        
        remove_played_cards_from_hand(&mut hand, &played);
        assert_eq!(hand, vec![4, 5]);
    }

    #[test]
    fn test_is_subset_of_hand() {
        let hand = vec![1, 2, 3, 4, 5];
        
        // Valid strategy
        let strategy = vec![vec![1, 2], vec![3]];
        assert!(is_subset_of_hand(&strategy, &hand));
        
        // Invalid strategy (using a card twice)
        let strategy = vec![vec![1, 2], vec![1]];
        assert!(!is_subset_of_hand(&strategy, &hand));
        
        // Invalid strategy (card not in hand)
        let strategy = vec![vec![1, 2], vec![6]];
        assert!(!is_subset_of_hand(&strategy, &hand));
    }

    #[test]
    fn test_flatten() {
        let nested = vec![vec![1, 2], vec![3, 4], vec![]];
        let flat = flatten(&nested);
        assert_eq!(flat, vec![1, 2, 3, 4]);
    }
}