use std::cmp::Ordering;
use std::iter::zip;
use rand::Rng;
use rand_chacha::ChaCha12Rng; // Import ChaCha12Rng

use crate::game_state::GameState;

use crate::game_types::{
    Actor, Reward, Cards, ScherbiusAction, TuringAction, Action, BattleOutcomeDetail,
};
use crate::enigma;
use crate::utils;

// Helper: Determines the winner of a single battle
pub fn battle_result(scherbius_cards: &Cards, turing_cards: &Cards) -> Option<Actor> {
    match scherbius_cards.iter().sum::<u32>().cmp(&turing_cards.iter().sum::<u32>()) {
        Ordering::Less => Some(Actor::Turing),
        Ordering::Greater => Some(Actor::Scherbius),
        Ordering::Equal => None,
    }
}

// Helper: Samples a single battle reward
pub fn sample_battle_reward(max_vp: u32, max_draw: u32, rng: &mut ChaCha12Rng) -> Reward { // Changed type
    // ... rest of function
    match rng.gen_range(0..2) {
        0 => {
            if max_vp == 0 { Reward::Null }
            else { Reward::VictoryPoints(rng.gen_range(1..=max_vp)) }
        }
        1 => {
            if max_draw == 0 { Reward::Null }
            else {
                let num_cards = rng.gen_range(1..=max_draw);
                Reward::NewCards(utils::draw_cards(num_cards, rng))
            }
        }
        _ => unreachable!(),
    }
}

// Helper: Generates rewards for all battles
pub fn random_rewards(n: u32, max_vp: u32, max_draw: u32, rng: &mut ChaCha12Rng) -> Vec<Reward> { // Changed type
    (0..n).map(|_| sample_battle_reward(max_vp, max_draw, rng)).collect()
}

// Logic for Scherbius intercepting strategy (mutates encoder in state)
pub fn intercept_scherbius_strategy(state: &GameState, strategy: &[Cards]) -> (enigma::EasyEnigma, Vec<Cards>) {
    let mut cloned_encoder = state.encoder.clone();
    let intercepted_cards_vector = strategy.iter().map(|h| cloned_encoder.call(h)).collect();
    (cloned_encoder, intercepted_cards_vector)
}

// Validates a player's action
pub fn check_action_validity(state: &GameState, action: &Action) -> Result<(), String> {
    let (player_name, current_strategy, hand) = match action {
        Action::TuringAction(turing_action) => ("Turing", &turing_action.strategy, &state.turing_hand),
        Action::ScherbiusAction(scherbius_action) => ("Scherbius", &scherbius_action.strategy, &state.scherbius_hand),
    };

    if current_strategy.len() as u32 != state.game_config.n_battles {
        return Err(format!(
            "{}: Strategy must cover all {} battles. Got: {}",
            player_name,
            state.game_config.n_battles,
            current_strategy.len()
        ));
    }

    if !utils::is_subset_of_hand(current_strategy, hand) {
        return Err(format!("{}: Strategy cards {:?} are not a subset of hand {:?}", player_name, current_strategy, hand));
    }

    for cards_in_battle in current_strategy {
        if cards_in_battle.len() as u32 > state.game_config.max_cards_per_battle {
            return Err(format!(
                "{}: Too many cards ({}) committed to a single battle. Max: {}",
                player_name,
                cards_in_battle.len(),
                state.game_config.max_cards_per_battle
            ));
        }
    }
    Ok(())
}

// The main game step logic function
pub fn process_step(
    mut current_state: GameState, // Takes ownership, can mutate this copy
    scherbius_action: &ScherbiusAction,
    turing_action: &TuringAction,
) -> Result<(GameState, Vec<BattleOutcomeDetail>), String> {
    // Validate actions
    check_action_validity(&current_state, &Action::ScherbiusAction(scherbius_action.clone()))?;
    check_action_validity(&current_state, &Action::TuringAction(turing_action.clone()))?;

    // Remove played cards from hands
    utils::remove_played_cards_from_hand(&mut current_state.scherbius_hand, &scherbius_action.strategy);
    utils::remove_played_cards_from_hand(&mut current_state.turing_hand, &turing_action.strategy);

    let mut detailed_outcomes: Vec<BattleOutcomeDetail> = Vec::new();

    // Process battles
    for (i, (s_cards, t_cards)) in zip(
        scherbius_action.strategy.iter(),
        turing_action.strategy.iter(),
    ).enumerate() {
        let t_sum = t_cards.iter().sum::<u32>();
        let s_sum = s_cards.iter().sum::<u32>();
        let winner_option = battle_result(s_cards, t_cards);
        let reward_for_this_battle = current_state.rewards[i].clone(); // Clone reward for storing

        let mut turing_cards_won_in_battle = Vec::new();
        let mut turing_vp_won_in_battle = 0;
        let mut scherbius_cards_won_in_battle = Vec::new();
        let mut scherbius_vp_won_in_battle = 0;

        let actual_battle_winner = winner_option.unwrap_or(Actor::Null); // Handle draw case

        if let Some(winner) = winner_option {
            // The reward_for_this_battle is applied based on who won
            match winner {
                Actor::Turing => match &reward_for_this_battle {
                    Reward::VictoryPoints(v) => {
                        current_state.turing_points += *v;
                        turing_vp_won_in_battle = *v;
                    }
                    Reward::NewCards(cards) => {
                        current_state.turing_hand.extend_from_slice(cards);
                        turing_cards_won_in_battle = cards.clone();
                    }
                    Reward::Null => (),
                },
                Actor::Scherbius => match &reward_for_this_battle {
                    Reward::VictoryPoints(v) => {
                        current_state.scherbius_points += *v;
                        scherbius_vp_won_in_battle = *v; // Track Scherbius VP
                    }
                    Reward::NewCards(cards) => {
                        current_state.scherbius_hand.extend_from_slice(cards);
                        scherbius_cards_won_in_battle = cards.clone(); // Track Scherbius cards
                    }
                    Reward::Null => (),
                },
                Actor::Null => (), // This case within Some(winner) should ideally not be hit if battle_result only returns Some(Scherbius), Some(Turing), or None.
            }
        }
        // If winner_option was None (a draw), no points/cards are awarded from the reward_for_this_battle.

        detailed_outcomes.push(BattleOutcomeDetail {
            battle_index: i as u32, // The current battle index
            scherbius_cards_played: s_cards.clone(), // Cards Scherbius played in this battle
            turing_cards_played: t_cards.clone(),    // Cards Turing played in this battle
            scherbius_sum: s_sum,
            turing_sum: t_sum,
            battle_winner: actual_battle_winner, // Winner of this specific battle
            reward_applied: reward_for_this_battle, // The reward that was up for grabs in this battle
            scherbius_vp_won: scherbius_vp_won_in_battle,
            scherbius_cards_won: scherbius_cards_won_in_battle,
            turing_vp_won: turing_vp_won_in_battle,
            turing_cards_won: turing_cards_won_in_battle,
        });
    }

    // Scherbius encryption action
    if scherbius_action.encryption {
        if current_state.scherbius_points >= current_state.game_config.encryption_cost {
            current_state.scherbius_points -= current_state.game_config.encryption_cost;
            current_state.encoder = enigma::EasyEnigma::new(
                current_state.game_config.encryption_vocab_size,
                current_state.game_config.encryption_k_rotors as usize,
                &mut current_state.rng, // Use rng from current_state
            );
            if current_state.game_config.verbose {
                println!("Scherbius re-encrypted. New Enigma settings generated.");
            }
        } else if current_state.game_config.verbose {
            println!("Scherbius attempted re-encryption but lacked points.");
        }
    }

    // Check for game winner
    if current_state.scherbius_points >= current_state.game_config.victory_points {
        current_state.winner = Actor::Scherbius;
    } else if current_state.turing_points >= current_state.game_config.victory_points {
        current_state.winner = Actor::Turing;
    }

    if current_state.winner != Actor::Null {
        return Ok((current_state, detailed_outcomes)); // Game over
    }

    // Prepare for next round: new rewards, deal cards
    current_state.rewards = random_rewards(
        current_state.game_config.n_battles,
        current_state.game_config.max_vp,
        current_state.game_config.max_draw,
        &mut current_state.rng,
    );

    let scherbius_new_cards = utils::draw_cards(current_state.game_config.scherbius_deal, &mut current_state.rng);
    current_state.scherbius_hand.extend_from_slice(&scherbius_new_cards);
    let turing_new_cards = utils::draw_cards(current_state.game_config.turing_deal, &mut current_state.rng);
    current_state.turing_hand.extend_from_slice(&turing_new_cards);

    // Enforce max hand size
    if current_state.scherbius_hand.len() > current_state.game_config.max_hand_size as usize {
        current_state.scherbius_hand.truncate(current_state.game_config.max_hand_size as usize);
    }
    if current_state.turing_hand.len() > current_state.game_config.max_hand_size as usize {
        current_state.turing_hand.truncate(current_state.game_config.max_hand_size as usize);
    }

    // Reset enigma steps for the next round
    current_state.encoder.reset_steps();

    Ok((current_state, detailed_outcomes))
}