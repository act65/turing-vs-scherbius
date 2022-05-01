use turing_vs_scherbius::play;
use turing_vs_scherbius::{GameState, GameConfig};

mod user_input;

fn main() {

    let game_config = GameConfig::new();
    let mut game_state = GameState::new(&game_config);
    println!("{}", game_state);

    play(
        game_config,
        game_state, 
        user_input::user_input_player, 
        user_input::user_input_player
    );
}