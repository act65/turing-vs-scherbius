use turing_vs_scherbius::play;
use turing_vs_scherbius::GameConfig;
use std::fs;

mod players;

fn read_config(path: String) -> GameConfig {
    let data = fs::read_to_string(path).expect("Unable to read file");
    let game_config: GameConfig = serde_json::from_str(&data).expect("JSON was not well-formatted");
    return game_config
}


fn play(
    game_config: GameConfig,
    sherbius: ScherbiusPlayer, 
    turing: TuringPlayer) -> Actor{

    let mut game_state = GameState::new(&game_config);

    loop {
        // Sherbius plays first
        let scherbius_action = sherbius(&game_state.scherbius_hand, &game_state.rewards);
        let intercepted_scherbius_strategy = game_state.intercept_scherbius_strategy(&scherbius_action.strategy);
        // Turing plays second
        let turing_action = turing(&game_state.turing_hand, &game_state.rewards, &intercepted_scherbius_strategy);

        game_state.step(
                &scherbius_action, 
                &turing_action);

        if game_config.verbose {
            println!("{:?}", game_state);
        }

        if game_state.winner != Actor::Null {
            break
        }
        
    }
    if game_config.verbose {
        println!("Winner: {:?}", game_state.winner);
    }
    game_state.winner
}

fn main() {
    let path: String = "config.json".to_string(); 
    let game_config: GameConfig = read_config(path);

    play(
        game_config,
        // players::scherbius_human_player,
        players::random_scherbius_player,
        // players::turing_human_player,
        players::random_turing_player
    );
}