use turing_vs_scherbius::play;
use turing_vs_scherbius::{GameState, GameConfig};

mod players;

fn main() {

    let game_config = GameConfig {
        scherbius_starting: 6,
        scherbius_deal: 3,
        scherbius_hand_limit: 4,
        turing_starting: 4,
        turing_deal: 2,
        turing_hand_limit: 4,
        victory_points: 15,
        n_battles: 3,
    };
    let mut game_state = GameState::new(&game_config);
    println!("{}", game_state);

    play(
        game_config,
        game_state, 
        players::user_input_player, 
        players::random_player
    );
}