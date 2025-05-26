use turing_vs_scherbius::{
    ScherbiusAction, 
    TuringAction, 
    Reward, 
    Cards, 
    EncryptionCode,
    utils
};

fn get_strategy_from_hand(hand: &[u32], rewards: &[Reward], prompt: &str) -> Vec<Cards> {
    rewards.iter().map(|r| {
        println!("Rewards: {:?}", r);
        println!("Hand {:?}", hand);
        utils::choose_from_set(hand, prompt)  // Fixed to pass a slice reference
    }).collect()
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

    let strategy_prompt: String = "Choose your strategy (enter to end)".to_string();
    let strategy = get_strategy_from_hand(&turing_hand, &rewards, &strategy_prompt);

    TuringAction {
        strategy: strategy,
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

    let encryption_prompt: String = "Choose whether to re-encrypt (0->False, 1->True)".to_string();
    let choice = utils::get_user_input(&encryption_prompt);
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

pub fn random_scherbius_player(
        scherbius_hand: &Vec<u32>, 
        rewards: &Vec<Reward>)
        ->ScherbiusAction{

    let mut hand = scherbius_hand.clone();
    let strategy = utils::get_rnd_strategy(&mut hand, rewards.len());
    let encrypt = utils::random_reencryption();

    ScherbiusAction {
        strategy: strategy,
        encryption: encrypt,
    }
}

pub fn random_turing_player(
    turing_hand: &Vec<u32>, 
    rewards: &Vec<Reward>,
    _encrypted_cards: &Vec<Cards>)
    ->TuringAction{

    let mut hand = turing_hand.clone();
    let strategy = utils::get_rnd_strategy(&mut hand, rewards.len());

    TuringAction {
        strategy: strategy,
    }
}