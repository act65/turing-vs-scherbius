use turing_vs_scherbius::play;
use turing_vs_scherbius::GameConfig;

mod players;
// use crate::players::{random_scherbius_player, random_turing_player};

fn main() {

    let game_config = GameConfig {
        scherbius_starting: 6,
        scherbius_deal: 4,

        turing_starting: 4,
        turing_deal: 3,

        victory_points: 100,
        n_battles: 10,

        encryption_cost: 25,
    };

    let winner = play(
        game_config,
        players::random_scherbius_player, 
        // players::random_player
        // players::user_input_player, 
        players::random_turing_player
    );
    println!("Winner: {:?}", winner);
}