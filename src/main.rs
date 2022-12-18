use turing_vs_scherbius::play;
use turing_vs_scherbius::GameConfig;
use std::fs;
// use serde_json::from_str;

mod players;
// use crate::players::{random_scherbius_player, random_turing_player};


// fn read_config(path: str) {
//     let data = fs::read_to_string(path).expect("Unable to read file");
//     let game_config: GameConfig = serde_json::from_str(the_file).expect("JSON was not well-formatted");
//     println!("{:?}", game_config)
// }

fn main() {

    // TODO change to using arrays instead of vecs.
    // can pad arrays with -1 for missing entries?

    // TODO fetch these config variables from env variables or a config file
    // let game_config =
    // static str path = "config.json"; 
    // read_config(path);

    let game_config = GameConfig {
        scherbius_starting: 6,
        scherbius_deal: 4,
        turing_starting: 4,
        turing_deal: 3,
        victory_points: 20,
        n_battles: 3,
        encryption_cost: 25,
        encryption_code_len: 1,
        verbose: true
        // TODO a nicer format for the verbose? something that is more easily parsed?
    };

    let winner = play(
        game_config,
        players::scherbius_human_player,
        // players::random_scherbius_player, 
        // players::turing_human_player,
        players::random_turing_player
    );
    println!("Winner: {:?}", winner);
}