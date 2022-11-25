use turing_vs_scherbius::play;
use turing_vs_scherbius::GameConfig;

mod players;
// use crate::players::{random_scherbius_player, random_turing_player};

fn main() {

    // TODO change to using arrays instead of vecs.
    // can pad arrays with -1 for missing entries?

    // TODO fetch these config variables from env variables or a config file
    let game_config = GameConfig {
        scherbius_starting: 6,
        scherbius_deal: 4,

        turing_starting: 4,
        turing_deal: 3,

        victory_points: 100,
        n_battles: 2,

        encryption_cost: 25,
    };

    let winner = play(
        game_config,
        // players::random_scherbius_player, 
        players::user_input_player, 
        players::random_turing_player
    );
    println!("Winner: {:?}", winner);
}