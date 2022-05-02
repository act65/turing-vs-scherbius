use std::fmt;
use std::cmp::Ordering;
use std::iter::zip;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

#[derive(Debug)]
pub struct GameConfig {
    pub scherbius_starting: u32,
    pub scherbius_deal: u32,
    pub turing_starting: u32,
    pub turing_deal: u32,
    pub victory_points: u32,
    pub n_battles: u32,

    pub encryption_cost: u32,
    // re-encrypting costs victory points?!

}

#[derive(Debug)]
pub struct GameState {
    pub turing_hand: Vec<u32>,
    pub scherbius_hand: Vec<u32>,

    pub encryption_broken: bool,
    pub encryption: (u32, u32),
    // could vary the number of values used for encryption?!

    pub turing_points: u32,
    pub scherbius_points: u32,
}

// initial game state
impl GameState {
    pub fn new(game_config: &GameConfig) -> GameState {
        let mut rng = rand::thread_rng(); 
        GameState{
            // deal inital random hands
            scherbius_hand: draw_cards(game_config.scherbius_starting),
            turing_hand: draw_cards(game_config.turing_starting),
        
            encryption_broken: false,
            encryption: (rng.gen_range(1..10), rng.gen_range(1..10)),

            turing_points: 0,
            scherbius_points: 0,
        }
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Turing: {:?}\nSherbius: {:?}", self.turing_hand, self.scherbius_hand)
    }
}

impl GameState{
    pub fn step(
        &mut self,
        game_config: &GameConfig, 
        scherbius_actions: &ScherbiusAction,
        turing_actions: &TuringAction,
        rewards: &Vec<Reward>) {

    // each player gets some new cards
    let new_cards = draw_cards(game_config.scherbius_deal);
    self.scherbius_hand.extend_from_slice(&new_cards);

    let new_cards = draw_cards(game_config.turing_deal);
    self.turing_hand.extend_from_slice(&new_cards);

    // resolve battles
    let results: Vec<_> = zip(scherbius_actions.strategy.iter(), turing_actions.strategy.iter())
        .map(|(a1, a2)|battle_result(a1, a2))
        // this shouldnt work?!
        .filter(|x| x.is_some())
        .map(|x| x.unwrap())
        .collect();

    // distribute the rewards
    for (result, reward) in zip(results, rewards) {
        match result {
            Actor::Turing => 
                {match reward {
                    Reward::VictoryPoints(v) => self.turing_points = self.turing_points + v,
                    Reward::NewCards(cards) => self.turing_hand.extend_from_slice(&cards),
                    _ => ()
                    }},
            Actor::Scherbius => 
                {match reward {
                    Reward::VictoryPoints(v) => self.scherbius_points = self.scherbius_points + v,
                    Reward::NewCards(cards) => self.scherbius_hand.extend_from_slice(&cards),
                    _ => ()
                }}
            _ => ()
        }

    }

    // update encryption
    // TODO

    }
}

pub type ScherbiusPlayer = fn(&GameState, &Vec<Reward>) -> ScherbiusAction;
pub type TuringPlayer = fn(&GameState, &Vec<Reward>, &Vec<Cards>) -> TuringAction;

// #[derive(Debug)]
// pub struct Cards {
//     value: Vec<u32>
// }
pub type Cards = Vec<u32>;

#[derive(Debug)]
pub enum Actor {
    Scherbius,
    Turing,
    // Scherbius(Player),
    // Turing(Player),
}


fn battle_result(
    scherbius_cards: &Cards, 
    turing_cards: &Cards) -> Option<Actor> {
    match (scherbius_cards.iter().sum::<u32>()).cmp(&turing_cards.iter().sum()) {
        Ordering::Less => Some(Actor::Turing),
        Ordering::Greater => Some(Actor::Scherbius),
        Ordering::Equal => None,
    }
}

#[derive(Debug)]
pub enum Reward {
    VictoryPoints(u32),
    NewCards(Vec<u32>),
    Null,
}

impl Distribution<Reward> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Reward {
        match rng.gen_range(0..2) {
            // TODO distribution should make larger values less likely
            0 => Reward::VictoryPoints(rng.gen_range(1..10)),
            1 => Reward::NewCards(draw_cards(rng.gen_range(1..10))),
            _ => Reward::Null,
        }
    }
}

fn random_rewards(n: u32)->Vec<Reward> {
    let mut rewards = Vec::new();

    for _ in 0..n {
        let reward: Reward = rand::random();
        rewards.push(reward)
    }
    rewards
}

#[derive(Debug)]
pub struct TuringAction {
    pub strategy: Vec<Cards>,
    pub guesses: Option<Vec<(u32, u32)>>,
}

#[derive(Debug)]
pub struct ScherbiusAction {
    pub strategy: Vec<Cards>,
    pub encryption: Option<(u32, u32)>,
}

fn draw_cards(n: u32)->Cards {
    let mut cards = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..n {
        let value: u32 = rng.gen_range(1..11);
        cards.push(value)
    }
    cards
}

// fn encrypt(cards: Vec<u32>)->Vec<u32> {
    
// }

pub fn play(
        game_config: GameConfig,
        mut game_state: GameState, 
        sherbius: ScherbiusPlayer, 
        turing: TuringPlayer) {

    loop {
        // what is being played for this round?
        let rewards = random_rewards(game_config.n_battles);

        // Sherbius plays first
        let scherbius_action = sherbius(&game_state, &rewards);
        // let encrypted_strategy = encrypt(&scherbius_action.strategy);
        let encrypted_strategy = scherbius_action.strategy.clone();

        // Turing plays second
        let turing_action = turing(&game_state, &rewards, &encrypted_strategy);
        // gamestate = gamestate.update(&action);

        // check_action_validity(turing_action);
        // check_action_validity(sherbius_action);

        game_state.step(
                &game_config,
                &scherbius_action, 
                &turing_action, 
                &rewards);

        // check if a player has won
        if game_state.scherbius_points >= game_config.victory_points {
            println!("Scherbius wins");
            break;}
        else if game_state.turing_points >= game_config.victory_points {
            println!("Turing wins");
            break;
        }
    }
}

// #[cfg(test)]
// mod tests {

//     #[test]
//     fn test_encrypt() {
//         let strategy = get_rnd_strategy();
//         let encryption = (0, 1);

//         assert_eq!();
//     }
// }