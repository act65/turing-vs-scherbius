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
    // re-encrypting costs victory points
}

#[derive(Debug)]
struct GameState {
    turing_hand: Vec<u32>,
    scherbius_hand: Vec<u32>,

    encryption_broken: bool,
    encryption: [u32; 2],
    // TODO want to vary the number of values used for encryption?!
    // 11^2 = 121. quite hard to break encryption!

    turing_points: u32,
    scherbius_points: u32,
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Turing: {:?}\nSherbius: {:?}", self.turing_hand, self.scherbius_hand)
    }
}

impl GameState{
    fn new(game_config: &GameConfig) -> GameState {
        let mut rng = rand::thread_rng(); 
        GameState{
            // deal inital random hands
            scherbius_hand: draw_cards(game_config.scherbius_starting),
            turing_hand: draw_cards(game_config.turing_starting),
        
            encryption_broken: false,
            encryption: [rng.gen_range(1..10), rng.gen_range(1..10)],

            turing_points: 0,
            scherbius_points: 0,
        }
    }

    fn step(
        &mut self,
        game_config: &GameConfig, 
        scherbius_actions: &ScherbiusAction,
        turing_actions: &TuringAction,
        rewards: &Vec<Reward>) {

    println!("{:?}", self.scherbius_hand);
    println!("{:?}", scherbius_actions);

    println!("{:?}", self.turing_hand);
    println!("{:?}", turing_actions);


    // each player gets some new cards
    let new_cards = draw_cards(game_config.scherbius_deal);
    self.scherbius_hand.extend_from_slice(&new_cards);

    let new_cards = draw_cards(game_config.turing_deal);
    self.turing_hand.extend_from_slice(&new_cards);

    // remove cards played from hands
    // TODO move into fn
    for c in scherbius_actions.strategy.iter() {
        for i in c.iter() {
            let index = self.scherbius_hand.iter().position(|y| *y == *i).unwrap();
            self.scherbius_hand.remove(index);    
        }
    }
    for c in turing_actions.strategy.iter() {
        for i in c.iter() {
            let index = self.turing_hand.iter().position(|y| *y == *i).unwrap();
            self.turing_hand.remove(index);    
        }
    }
    // remove guesses from turing's hand
    for g in turing_actions.guesses.iter() {
        for i in g.iter() {
            let index = self.turing_hand.iter().position(|y| *y == *i).unwrap();
            self.turing_hand.remove(index);        
        }
    }

    // resolve battles
    let results: Vec<_> = zip(scherbius_actions.strategy.iter(), turing_actions.strategy.iter())
        .map(|(a1, a2)|battle_result(a1, a2))
        // this shouldnt work?!
        .filter(|x| x.is_some())
        .map(|x| x.unwrap())
        .collect();

    // TODO test that;
    // - draw means no one wins
    // - no cards vs cards means cards wins

    println!("{:?}", rewards);
    println!("{:?}", results);

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

    // resolve encryption guess
    for g in turing_actions.guesses.iter() {
        if g == &self.encryption {self.encryption_broken=true}
    }

    // reset encryption?
    if self.scherbius_points >= game_config.encryption_cost && scherbius_actions.encryption 
        {self.encryption = [0, 1];  // TODO randomly pick new encryption 
        self.encryption_broken=false};
    }

}

pub type ScherbiusPlayer = fn(&Vec<u32>, &Vec<Reward>) -> ScherbiusAction;
pub type TuringPlayer = fn(&Vec<u32>, &Vec<Reward>, &Vec<Cards>) -> TuringAction;

// #[derive(Debug)]
// pub struct Cards {
//     value: Vec<u32>
// }
pub type Cards = Vec<u32>;

#[derive(Debug)]
pub enum Actor {
    Scherbius,
    Turing,
    Null
    // Scherbius(Player),
    // Turing(Player),
}


fn battle_result(
    scherbius_cards: &Cards, 
    turing_cards: &Cards) -> Option<Actor> {
    match (scherbius_cards.iter().sum::<u32>()).cmp(&turing_cards.iter().sum::<u32>()) {
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
            // TODO want a parameter to control the max value in GameConfig?!
            // TODO move rng to be an arg?
            0 => Reward::VictoryPoints(rng.gen_range(1..5)),
            1 => Reward::NewCards(draw_cards(rng.gen_range(1..5))),
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
    pub guesses: Vec<[u32; 2]>,
}

#[derive(Debug)]
pub struct ScherbiusAction {
    pub strategy: Vec<Cards>,
    pub encryption: bool,
}

fn draw_cards(n: u32)->Cards {
    let mut cards = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let value: u32 = rng.gen_range(1..11);
        cards.push(value)
    }
    cards
}

// fn encrypt(cards: Vec<u32>)->Vec<u32> {
    
// }

pub fn play(
        game_config: GameConfig,
        sherbius: ScherbiusPlayer, 
        turing: TuringPlayer) -> Actor{

        let mut game_state = GameState::new(&game_config);
        let mut winner: Actor = Actor::Null;

    loop {
        // what is being played for this round?
        let rewards = random_rewards(game_config.n_battles);

        // Sherbius plays first
        let scherbius_action = sherbius(&game_state.scherbius_hand, &rewards);
        // let encrypted_strategy = encrypt(&scherbius_action.strategy);
        let intercepted_scherbius_strategy = if game_state.encryption_broken {scherbius_action.strategy.clone()}
            else {scherbius_action.strategy.clone()};

        // Turing plays second
        let turing_action = turing(&game_state.turing_hand, &rewards, &intercepted_scherbius_strategy);

        // check_action_validity(turing_action);
        // check_action_validity(sherbius_action);

        game_state.step(
                &game_config,
                &scherbius_action, 
                &turing_action, 
                &rewards);

        // check if a player has won
        if game_state.scherbius_points >= game_config.victory_points {
            winner = Actor::Scherbius;
            break;}
        else if game_state.turing_points >= game_config.victory_points {
            winner = Actor::Turing;
            break;
        }
    }
    winner
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