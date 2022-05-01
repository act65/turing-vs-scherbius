use turing_vs_scherbius::play;
use turing_vs_scherbius::GameState;
use turing_vs_scherbius::user_input_player;


fn main() {

    let gamestate = GameState::new();
    println!("{:?}", gamestate);

    play(gamestate, user_input_player, user_input_player);
}