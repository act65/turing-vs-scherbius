use turing_vs_scherbius::{
    play,
    GameConfig,
    read_config
}

mod players;

fn main() {
    let path: String = "config.json".to_string(); 
    let game_config: GameConfig = read_config(path);

    let winner = play(
        game_config,
        // players::scherbius_human_player,
        players::random_scherbius_player,
        // players::turing_human_player,
        players::random_turing_player
    );
}