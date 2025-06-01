use serde::{Serialize, Deserialize};
use pyo3::prelude::*;
use pyo3::types::PyTuple; // For Reward conversion


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

// Player function type aliases (can also be in lib.rs or where they are primarily used)
pub type ScherbiusPlayer = fn(&Vec<u32>, &Vec<Reward>) -> ScherbiusAction;
pub type TuringPlayer = fn(&Vec<u32>, &Vec<Reward>, &Vec<Cards>) -> TuringAction;

// src/game_types.rs



#[pyclass(clone)] // Keep clone for Py::new
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)] // serde for potential raw state, PartialEq for tests
pub struct BattleOutcomeDetail {
    // Public fields for Rust internal use
    pub battle_index: u32,
    pub scherbius_cards_played: Cards,
    pub turing_cards_played: Cards,
    pub scherbius_sum: u32,
    pub turing_sum: u32,
    pub battle_winner: Actor, // Store raw Actor enum
    pub reward_applied: Reward, // Store raw Reward enum
    pub scherbius_vp_won: u32,
    pub scherbius_cards_won: Cards,
    pub turing_vp_won: u32,
    pub turing_cards_won: Cards,
}

#[pymethods]
impl BattleOutcomeDetail {
    #[getter]
    fn battle_index(&self) -> u32 { self.battle_index }
    #[getter]
    fn scherbius_cards_played(&self) -> Cards { self.scherbius_cards_played.clone() }
    #[getter]
    fn turing_cards_played(&self) -> Cards { self.turing_cards_played.clone() }
    #[getter]
    fn scherbius_sum(&self) -> u32 { self.scherbius_sum }
    #[getter]
    fn turing_sum(&self) -> u32 { self.turing_sum }
    #[getter]
    fn scherbius_vp_won(&self) -> u32 { self.scherbius_vp_won }
    #[getter]
    fn scherbius_cards_won(&self) -> Cards { self.scherbius_cards_won.clone() }
    #[getter]
    fn turing_vp_won(&self) -> u32 { self.turing_vp_won }
    #[getter]
    fn turing_cards_won(&self) -> Cards { self.turing_cards_won.clone() }

    #[getter]
    fn battle_winner(&self) -> String {
        match self.battle_winner {
            Actor::Scherbius => "Scherbius".to_string(),
            Actor::Turing => "Turing".to_string(),
            Actor::Null => "Null".to_string(),
        }
    }

    #[getter]
    fn reward_applied(&self, py: Python) -> PyObject {
        match &self.reward_applied {
            Reward::VictoryPoints(v) => (0, *v).to_object(py), // Tuple: (type_id, value)
            Reward::NewCards(cards) => (1, cards.clone()).to_object(py),
            Reward::Null => (2, py.None()).to_object(py),
        }
    }
}