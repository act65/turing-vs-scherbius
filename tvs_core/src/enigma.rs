use rand::Rng;
use rand::seq::SliceRandom; // For shuffling to create permutations

/// Generates a random permutation of numbers from 0 to n-1.
fn generate_permutation(n: u32, rng: &mut impl Rng) -> Vec<u32> {
    let mut perm: Vec<u32> = (0..n).collect();
    perm.shuffle(rng);
    perm
}

/// Generates an inverse of a permutation.
fn invert_permutation(perm: &[u32]) -> Vec<u32> {
    let n = perm.len();
    let mut inv_perm = vec![0; n];
    for i in 0..n {
        inv_perm[perm[i] as usize] = i as u32;
    }
    inv_perm
}

/// Generates a valid reflector wiring (involution with no fixed points).
/// Pairs (0,1), (2,3), ...
/// If n is odd, the last element maps to itself, which is not ideal for Enigma reflector.
/// A better reflector ensures no char maps to itself.
fn generate_reflector(n: u32, rng: &mut impl Rng) -> Vec<u32> {
    if n % 2 != 0 {
        // For simplicity in this example, we'll require n to be even for a perfect reflector.
        // Or, handle the odd case by making the last element map to itself, though not ideal.
        // A true Enigma reflector would always have n=26 (even).
        // For this example, let's make a simple reflector:
        // 0<>1, 2<>3, etc. If n is odd, n-1 maps to itself.
        // This is NOT a cryptographically strong way to generate a reflector,
        // but suffices for demonstration of the *concept*.
        // A real Enigma reflector had fixed, carefully chosen pairings.
        let mut reflector = vec![0; n as usize];
        let mut p: Vec<u32> = (0..n).collect();
        p.shuffle(rng); // Shuffle to make pairings random

        for i in (0..n).step_by(2) {
            if i + 1 < n {
                reflector[p[i as usize] as usize] = p[(i + 1) as usize];
                reflector[p[(i + 1) as usize] as usize] = p[i as usize];
            } else {
                // Odd n, last element maps to itself (Enigma flaw if it happened)
                reflector[p[i as usize] as usize] = p[i as usize];
            }
        }
        return reflector;
    }

    // For even n, ensure no fixed points
    let mut p: Vec<u32> = (0..n).collect();
    p.shuffle(rng); // Shuffle to make pairings random
    let mut reflector = vec![0u32; n as usize];
    for i in (0..p.len()).step_by(2) {
        reflector[p[i] as usize] = p[i+1];
        reflector[p[i+1] as usize] = p[i];
    }
    reflector
}


#[derive(Debug, Clone)]
pub struct EasyEnigma {
    rotor_wirings: [Vec<u32>; 2],
    rotor_inv_wirings: [Vec<u32>; 2],
    reflector_wiring: Vec<u32>,
    step: [u32; 2], // Current rotational offset
    n: u32,
}

impl EasyEnigma {
    pub fn new(n: u32, rng: &mut impl Rng) -> Self {
        if n == 0 { panic!("n cannot be 0"); }
        // For simplicity, we'll often assume n=26 for letter mapping.
        // Real Enigma rotors were specific permutations. Here we generate random ones.
        let r0_wiring = generate_permutation(n, rng);
        let r1_wiring = generate_permutation(n, rng);
        let r0_inv_wiring = invert_permutation(&r0_wiring);
        let r1_inv_wiring = invert_permutation(&r1_wiring);

        let reflector = generate_reflector(n, rng);

        Self {
            rotor_wirings: [r0_wiring, r1_wiring],
            rotor_inv_wirings: [r0_inv_wiring, r1_inv_wiring],
            reflector_wiring: reflector,
            step: [0, 0], // Initial positions
            n,
        }
    }

    /// Helper function to pass a signal through a rotor (forward or backward)
    fn pass_rotor(val: u32, rotor_idx: usize, step_val: u32, forward: bool, n: u32,
                  wirings: &[Vec<u32>], inv_wirings: &[Vec<u32>]) -> u32 {
        let effective_input = (val + step_val) % n;
        let wired_output = if forward {
            wirings[rotor_idx][effective_input as usize]
        } else {
            inv_wirings[rotor_idx][effective_input as usize]
        };
        (wired_output + n - step_val) % n // Add n before -step_val to handle underflow
    }
    
    pub fn call_char(&mut self, x: u32) -> u32 {
        // The historical Enigma usually stepped *before* encryption for the current letter.
        // However, your original EasyEnigma stepped *after*. Let's stick to after for now.
        // If stepping before:
        // self.step[0] = (self.step[0] + 1) % self.n;
        // if self.step[0] == 0 { // Assuming step[0] just turned over
        //     self.step[1] = (self.step[1] + 1) % self.n;
        // }

        // Forward path
        let mut current_val = x;
        current_val = Self::pass_rotor(current_val, 0, self.step[0], true, self.n, &self.rotor_wirings, &self.rotor_inv_wirings);
        current_val = Self::pass_rotor(current_val, 1, self.step[1], true, self.n, &self.rotor_wirings, &self.rotor_inv_wirings);

        // Reflector
        current_val = self.reflector_wiring[current_val as usize];

        // Backward path
        current_val = Self::pass_rotor(current_val, 1, self.step[1], false, self.n, &self.rotor_wirings, &self.rotor_inv_wirings);
        current_val = Self::pass_rotor(current_val, 0, self.step[0], false, self.n, &self.rotor_wirings, &self.rotor_inv_wirings);
        
        let encrypted_char = current_val;

        // Step for the next character (as in your original design)
        self.step[0] = (self.step[0] + 1) % self.n;
        if self.step[0] == 0 {
            self.step[1] = (self.step[1] + 1) % self.n;
        }
        
        encrypted_char
    }


    pub fn call(&mut self, array: &Vec<u32>) -> Vec<u32> {
        let mut encrypted_array = Vec::with_capacity(array.len());
        for &x_val in array {
            encrypted_array.push(self.call_char(x_val));
        }
        encrypted_array
    }

    pub fn reset_steps(&mut self) {
        self.step = [0, 0];
    }
    
    /// Sets the rotor initial step positions
    pub fn set_steps(&mut self, steps: [u32; 2]) {
        self.step[0] = steps[0] % self.n;
        self.step[1] = steps[1] % self.n;
    }

    /// Sets the rotor wirings and reflector. This is a more complex "set" operation.
    pub fn set_wirings(&mut self, r0_wiring: Vec<u32>, r1_wiring: Vec<u32>, reflector_wiring: Vec<u32>) {
        assert_eq!(r0_wiring.len() as u32, self.n, "Rotor 0 wiring length mismatch");
        assert_eq!(r1_wiring.len() as u32, self.n, "Rotor 1 wiring length mismatch");
        assert_eq!(reflector_wiring.len() as u32, self.n, "Reflector wiring length mismatch");
        // TODO: Add validation for permutation and reflector properties

        self.rotor_wirings = [r0_wiring.clone(), r1_wiring.clone()];
        self.rotor_inv_wirings = [invert_permutation(&r0_wiring), invert_permutation(&r1_wiring)];
        self.reflector_wiring = reflector_wiring;
    }

    #[cfg(test)]
    pub fn get_steps(&self) -> [u32; 2] {
        self.step
    }
    #[cfg(test)]
    pub fn get_rotor_wirings(&self) -> &[Vec<u32>; 2] {
        &self.rotor_wirings
    }
    #[cfg(test)]
    pub fn get_reflector_wiring(&self) -> &Vec<u32> {
        &self.reflector_wiring
    }
}


#[cfg(test)]
mod true_enigma_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_permutation_inversion() {
        let perm = vec![2, 0, 1, 3]; // 0->2, 1->0, 2->1, 3->3
        let inv = invert_permutation(&perm);
        assert_eq!(inv, vec![1, 2, 0, 3]); // 0->1, 1->2, 2->0, 3->3
    }

    #[test]
    fn test_reflector_generation_even() {
        let n = 4;
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let reflector = generate_reflector(n, &mut rng);
        // Check properties:
        // 1. Self-inverse: reflector[reflector[i]] == i
        // 2. No fixed points: reflector[i] != i
        for i in 0..n {
            assert_eq!(reflector[reflector[i as usize] as usize], i, "Not self-inverse at {}", i);
            assert_ne!(reflector[i as usize], i, "Fixed point at {}", i);
        }
    }
    
    #[test]
    fn test_reflector_generation_odd_exception() {
        // Our simple odd generator might have one fixed point if not careful
        // A better one would disallow odd n or use a standard like UKW-D for Enigma M4 (which was n=26)
        // For now, let's test that it works for even n.
        // If you must support odd n, the reflector generation needs more thought for "no fixed points".
        // My current generate_reflector for odd n will have one fixed point.
        let n = 3;
        let seed = [1u8; 32]; // Different seed
        let mut rng = StdRng::from_seed(seed);
        let reflector = generate_reflector(n, &mut rng);
        let mut fixed_points = 0;
        for i in 0..n {
            assert_eq!(reflector[reflector[i as usize] as usize], i, "Not self-inverse at {}", i);
            if reflector[i as usize] == i {
                fixed_points += 1;
            }
        }
        // With n=3, exactly one element is paired with itself in the random shuffle pairing logic
        assert_eq!(fixed_points, 1, "Expected one fixed point for odd n with this generator");
    }

    #[test]
    fn test_true_easy_enigma_encryption_self_reciprocal() {
        let n = 26; // Standard alphabet size
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(n, &mut rng);

        let plaintext_char = 5; // 'f'
        let ciphertext_char = enigma.call_char(plaintext_char);

        // Property of reflector: no char encrypts to itself
        assert_ne!(plaintext_char, ciphertext_char, "A char should not encrypt to itself with a proper reflector.");

        // Reset steps and encrypt the ciphertext_char
        enigma.reset_steps(); // CRITICAL for testing self-reciprocity from same start state
        let decrypted_char = enigma.call_char(ciphertext_char);

        assert_eq!(plaintext_char, decrypted_char, "Enigma should be self-reciprocal.");
    }

    #[test]
    fn test_true_easy_enigma_stepping_changes_output() {
        let n = 26;
        let seed = [1u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(n, &mut rng);
        
        let input_val = 10; // 'k'
        
        let first_encrypted = enigma.call_char(input_val);
        // enigma.step has now advanced
        
        let second_encrypted = enigma.call_char(input_val); // Same input, but steps are different
        
        assert_ne!(first_encrypted, second_encrypted, "Output should change due to rotor stepping.");
    }

    #[test]
    fn test_true_easy_enigma_reset_steps() {
        let n = 26;
        let seed = [2u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(n, &mut rng);
        
        let input_array = vec![1, 2, 3, 4, 5];
        let _ = enigma.call(&input_array); // Encrypt to advance steps
        
        let steps_after_call = enigma.get_steps();
        assert_ne!(steps_after_call, [0,0], "Steps should have advanced.");

        enigma.reset_steps();
        let steps_after_reset = enigma.get_steps();
        assert_eq!(steps_after_reset, [0,0], "Steps should be reset to [0,0].");

        // Encrypting the same array again should produce the same result as if from start
        let mut enigma2 = EasyEnigma::new(n, &mut rng); // Fresh enigma with same seed will have same wirings
        // To make it truly identical, we need to ensure enigma2 has the same wirings as enigma1 did *initially*
        // This requires either passing the wirings or ensuring new() with same seed is identical.
        // For this test, let's re-use enigma and encrypt again after reset.
        
        let mut enigma_fresh_for_comparison = EasyEnigma {
            rotor_wirings: enigma.rotor_wirings.clone(),
            rotor_inv_wirings: enigma.rotor_inv_wirings.clone(),
            reflector_wiring: enigma.reflector_wiring.clone(),
            step: [0,0],
            n: enigma.n,
        };

        let first_run_output = enigma_fresh_for_comparison.call(&input_array);
        
        enigma.reset_steps(); // enigma was already used, reset it
        let second_run_output_after_reset = enigma.call(&input_array);

        assert_eq!(first_run_output, second_run_output_after_reset, "Encryption after reset should match initial encryption.");
    }
}