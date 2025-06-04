use serde::{Serialize, Deserialize};
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GameConfig {
    pub scherbius_starting: u32,
    pub turing_starting: u32,
    pub scherbius_deal: u32,
    pub turing_deal: u32,
    pub victory_points: u32,
    pub n_battles: u32,
    pub encryption_cost: u32,
    pub encryption_vocab_size: u32,
    pub encryption_k_rotors: u32,
    pub max_vp: u32,
    pub max_draw: u32,
    pub verbose: bool,
    pub max_hand_size: u32,
    pub max_cards_per_battle: u32,
}

#[pyclass(name = "GameConfig")] // Explicit Python name
#[derive(Clone)]
pub struct PyGameConfig {
    // Keep Arc<GameConfig> internally for easy conversion to Rust GameConfig
    pub inner: Arc<GameConfig>,
}

#[pymethods]
impl PyGameConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scherbius_starting: u32,
        scherbius_deal: u32,
        turing_starting: u32,
        turing_deal: u32,
        victory_points: u32,
        n_battles: u32,
        encryption_cost: u32,
        encryption_vocab_size: u32,
        encryption_k_rotors: u32,
        verbose: bool,
        max_vp: u32,
        max_draw: u32,
        max_hand_size: u32,
        max_cards_per_battle: u32,
    ) -> Self {
        PyGameConfig {
            inner: Arc::new(GameConfig {
                scherbius_starting,
                scherbius_deal,
                turing_starting,
                turing_deal,
                victory_points,
                n_battles,
                encryption_cost,
                encryption_vocab_size,
                encryption_k_rotors,
                verbose,
                max_vp,
                max_draw,
                max_hand_size,
                max_cards_per_battle,
            }),
        }
    }

    // Expose fields via getters if needed, or allow direct access if PyGameConfig fields are pub
    // For simplicity, if Python side just creates it, direct field access in Rust is fine.
    // If Python needs to read these after creation, add #[pyo3(get)] to GameConfig fields
    // and expose them through PyGameConfig if PyGameConfig fields are not pub.
    // The original code had #[pyo3(get)] on PyGameConfig fields, which is fine if they are distinct.
    // Here, PyGameConfig wraps an Arc<GameConfig>, so getters would access `self.inner.field`.
    // Let's add getters for consistency with the original PyGameConfig.

    #[getter] fn scherbius_starting(&self) -> u32 { self.inner.scherbius_starting }
    #[getter] fn scherbius_deal(&self) -> u32 { self.inner.scherbius_deal }
    #[getter] fn turing_starting(&self) -> u32 { self.inner.turing_starting }
    #[getter] fn turing_deal(&self) -> u32 { self.inner.turing_deal }
    #[getter] fn victory_points(&self) -> u32 { self.inner.victory_points }
    #[getter] fn n_battles(&self) -> u32 { self.inner.n_battles }
    #[getter] fn encryption_cost(&self) -> u32 { self.inner.encryption_cost }
    #[getter] fn encryption_vocab_size(&self) -> u32 { self.inner.encryption_vocab_size }
    #[getter] fn encryption_k_rotors(&self) -> u32 { self.inner.encryption_k_rotors }
    #[getter] fn verbose(&self) -> bool { self.inner.verbose }
    #[getter] fn max_vp(&self) -> u32 { self.inner.max_vp }
    #[getter] fn max_draw(&self) -> u32 { self.inner.max_draw }
    #[getter] fn max_hand_size(&self) -> u32 { self.inner.max_hand_size }
    #[getter] fn max_cards_per_battle(&self) -> u32 { self.inner.max_cards_per_battle }
}