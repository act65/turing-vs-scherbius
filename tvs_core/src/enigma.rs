use rand::Rng;

/// A simplified Enigma machine with two rotors
#[derive(Debug, Clone)]
pub struct EasyEnigma {
    rotor: [u32; 2],
    step: [u32; 2],
    n: u32,
}

impl EasyEnigma {
    /// Create a new EasyEnigma with randomized rotors
    pub fn new(n: u32, rng: &mut impl Rng) -> Self {
        Self {
            rotor: [rng.gen_range(1..n), rng.gen_range(1..n)],
            step: [0, 0],
            n,
        }
    }

    /// Encrypt an array of numbers
    pub fn call(&mut self, array: &Vec<u32>) -> Vec<u32> {
        let mut encrypted_array = Vec::with_capacity(array.len());

        for x in array {
            let y = (*x % (self.rotor[0] + self.step[0] + 1)) % (self.rotor[1] + self.step[1] + 1);
            encrypted_array.push(y);

            self.step[0] = (self.step[0] + 1) % self.n;
            if self.step[0] == 0 {
                self.step[1] = (self.step[1] + 1) % self.n;
            }
        }

        encrypted_array
    }

    /// Reset the steps back to initial position
    pub fn reset(&mut self) {
        self.step = [0, 0];
    }

    /// Set the rotor values
    pub fn set(&mut self, x: [u32; 2]) {
        self.rotor = x;
    }

    #[cfg(test)]
    pub fn get_rotors(&self) -> [u32; 2] {
        self.rotor
    }
}

// ... existing code ...

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_enigma_encryption() {
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(10, &mut rng);
        
        // Test encryption
        let input = vec![1, 2, 3, 4, 5];
        let encrypted = enigma.call(&input);
        
        // Ensure output is different from input
        assert_ne!(input, encrypted);
        
        // Ensure output length matches input
        assert_eq!(input.len(), encrypted.len());
    }

    #[test]
    fn test_enigma_reset() {
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(10, &mut rng);
        
        // Encrypt some values to advance the state
        let _ = enigma.call(&vec![1, 2, 3]);
        
        // Reset and encrypt again
        enigma.reset();
        let first_run = enigma.call(&vec![1, 2, 3]);
        
        // Reset and encrypt again - should be identical
        enigma.reset();
        let second_run = enigma.call(&vec![1, 2, 3]);
        
        assert_eq!(first_run, second_run);
    }

    #[test]
    fn test_enigma_set() {
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(10, &mut rng);
        
        // Set specific rotor values
        enigma.set([3, 7]);
        
        // Check that encryption behaves as expected
        let input = vec![1, 2, 3];
        let first_result = enigma.call(&input);
        
        // Reset enigma and change rotor values significantly
        enigma.reset();
        enigma.set([8, 2]); // Use very different values to ensure different encryption
        let second_result = enigma.call(&input.clone());
        
        // The two results should be different with different rotor settings
        assert_ne!(first_result, second_result);
    }
}