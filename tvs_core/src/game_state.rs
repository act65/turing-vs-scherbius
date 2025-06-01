// src/game_state.rs
use std::sync::Arc;
use rand::{Rng, SeedableRng}; // Keep SeedableRng
// Remove: use rand::rngs::StdRng;
use rand_chacha::ChaCha12Rng; // Import ChaCha12Rng
use serde::{Serialize, Deserialize};

use crate::enigma;
use crate::utils;
use crate::game_config::GameConfig;
use crate::game_types::{Actor, Reward, Cards};
use crate::game_logic::random_rewards;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub turing_hand: Vec<u32>,
    pub scherbius_hand: Vec<u32>,

    pub turing_points: u32,
    pub scherbius_points: u32,

    pub encoder: enigma::EasyEnigma,
    pub winner: Actor,

    pub rng: ChaCha12Rng, // Change StdRng to ChaCha12Rng

    pub rewards: Vec<Reward>,
    pub game_config: Arc<GameConfig>,
}

impl GameState {
    pub fn new(game_config: Arc<GameConfig>, seed: Option<u64>) -> GameState {
        let mut rng: ChaCha12Rng = match seed { // Explicitly type rng here
            Some(s) => ChaCha12Rng::seed_from_u64(s),
            None => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    // StdRng::from_rng(rand::thread_rng()) can be tricky.
                    // Let's seed ChaCha12Rng from thread_rng directly.
                    let mut thread_rng_seed = [0u8; 32];
                    rand::thread_rng().fill(&mut thread_rng_seed);
                    ChaCha12Rng::from_seed(thread_rng_seed)
                }
                #[cfg(target_arch = "wasm32")]
                {
                    // For wasm32, rand::thread_rng() often uses getrandom which might point to a weaker source
                    // or require specific JS shims. Seeding from its output is generally okay.
                    ChaCha12Rng::from_seed(rand::thread_rng().gen::<[u8; 32]>())
                }
            }
        };

        let rewards = random_rewards(
            game_config.n_battles,
            game_config.max_vp,
            game_config.max_draw,
            &mut rng,
        );

        GameState {
            scherbius_hand: utils::draw_cards(game_config.scherbius_starting, &mut rng),
            turing_hand: utils::draw_cards(game_config.turing_starting, &mut rng),
            turing_points: 0,
            scherbius_points: 0,
            encoder: enigma::EasyEnigma::new(
                game_config.encryption_vocab_size,
                game_config.encryption_k_rotors as usize,
                &mut rng,
            ),
            winner: Actor::Null,
            rng, // Store the ChaCha12Rng
            rewards,
            game_config,
        }
    }
}