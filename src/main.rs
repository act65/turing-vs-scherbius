use turing_vs_scherbius::play;
use turing_vs_scherbius::{GameState, GameConfig};

mod players;

fn main() {

    let game_config = GameConfig {
        scherbius_starting: 6,
        scherbius_deal: 3,

        turing_starting: 4,
        turing_deal: 2,

        victory_points: 15,
        n_battles: 8,

        encryption_cost: 5,
    };
    let mut game_state = GameState::new(&game_config);

    play(
        game_config,
        game_state, 
        players::random_scherbius_player, 
        // players::random_player
        // players::user_input_player, 
        players::random_turing_player
    );
}