use serde::{Serialize, Deserialize};
use pyo3::prelude::*;

pub type Cards = Vec<u32>;
// pub type EncryptionCode = Vec<u32>; // This type seems unused

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum Actor {
    Scherbius,
    Turing,
    Null,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Reward {
    VictoryPoints(u32),
    NewCards(Vec<u32>),
    Null,
}

#[derive(Debug, Clone)]
pub struct TuringAction {
    pub strategy: Vec<Cards>,
}

#[derive(Debug, Clone)]
pub struct ScherbiusAction {
    pub strategy: Vec<Cards>,
    pub encryption: bool,
}

// Enum to wrap specific actions, useful for validation
#[derive(Debug, Clone)]
pub enum Action {
    TuringAction(TuringAction),
    ScherbiusAction(ScherbiusAction),
}

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleOutcomeDetail {
    #[pyo3(get)]
    pub turing_sum: u32,
    #[pyo3(get)]
    pub scherbius_sum: u32,
    #[pyo3(get)]
    pub turing_cards_won: Vec<u32>,
    #[pyo3(get)]
    pub turing_vp_won: u32,
}

#[pymethods]
impl BattleOutcomeDetail {
    // No #[new] needed if it's only created in Rust and passed to Python
    // If Python needs to create it, add a #[new] method.
}

// Player function type aliases (can also be in lib.rs or where they are primarily used)
pub type ScherbiusPlayer = fn(&Vec<u32>, &Vec<Reward>) -> ScherbiusAction;
pub type TuringPlayer = fn(&Vec<u32>, &Vec<Reward>, &Vec<Cards>) -> TuringAction;