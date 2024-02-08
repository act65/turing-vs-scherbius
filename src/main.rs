use turing_vs_scherbius::play;
use turing_vs_scherbius::GameConfig;
use std::fs;
use serde_json::from_str;

mod players;

fn read_config(path: String) -> GameConfig {
    let data = fs::read_to_string(path).expect("Unable to read file");
    let game_config: GameConfig = serde_json::from_str(&data).expect("JSON was not well-formatted");
    return game_config
}

fn main() {
    let path: String = "config.json".to_string(); 
    let game_config: GameConfig = read_config(path);

    let winner = play(
        game_config,
        // players::scherbius_human_player,
        players::random_scherbius_player,
        players::turing_human_player,
        // players::random_turing_player
    );
}