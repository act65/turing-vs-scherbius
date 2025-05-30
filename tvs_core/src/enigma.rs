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

/// Generates a valid reflector wiring (involution with no fixed points for even n).
/// If n is odd, the current implementation might result in one fixed point.
fn generate_reflector(n: u32, rng: &mut impl Rng) -> Vec<u32> {
    if n % 2 != 0 {
        // For odd n, this simple generator pairs elements randomly.
        // If n is odd, one element will be left over and paired with itself.
        let mut reflector = vec![0; n as usize];
        let mut p: Vec<u32> = (0..n).collect();
        p.shuffle(rng); 

        for i in (0..n).step_by(2) {
            if i + 1 < n {
                reflector[p[i as usize] as usize] = p[(i + 1) as usize];
                reflector[p[(i + 1) as usize] as usize] = p[i as usize];
            } else {
                // Odd n, last element maps to itself
                reflector[p[i as usize] as usize] = p[i as usize];
            }
        }
        return reflector;
    }

    // For even n, ensure no fixed points by pairing distinct elements.
    let mut p: Vec<u32> = (0..n).collect();
    p.shuffle(rng); 
    let mut reflector = vec![0u32; n as usize];
    for i in (0..p.len()).step_by(2) {
        reflector[p[i] as usize] = p[i+1];
        reflector[p[i+1] as usize] = p[i];
    }
    reflector
}


#[derive(Debug, Clone)]
pub struct EasyEnigma {
    rotor_wirings: Vec<Vec<u32>>,     // Generalized from array to Vec
    rotor_inv_wirings: Vec<Vec<u32>>, // Generalized from array to Vec
    reflector_wiring: Vec<u32>,
    step: Vec<u32>,                   // Generalized from array to Vec
    n: u32,                           // vocabulary size
    k: usize,                         // number of rotors
}

impl EasyEnigma {
    pub fn new(n: u32, k: usize, rng: &mut impl Rng) -> Self { // Added k argument
        if n == 0 { panic!("n cannot be 0"); }
        if k == 0 { panic!("k (number of rotors) cannot be 0"); }

        let mut rotor_wirings = Vec::with_capacity(k);
        let mut rotor_inv_wirings = Vec::with_capacity(k);

        for _ in 0..k {
            let wiring = generate_permutation(n, rng);
            let inv_wiring = invert_permutation(&wiring);
            rotor_wirings.push(wiring);
            rotor_inv_wirings.push(inv_wiring);
        }

        let reflector = generate_reflector(n, rng);

        Self {
            rotor_wirings,
            rotor_inv_wirings,
            reflector_wiring: reflector,
            step: vec![0; k], // Initial positions for k rotors
            n,
            k, // Store k
        }
    }

    /// Helper function to pass a signal through a specific rotor (forward or backward)
    fn pass_rotor_through_idx(&self, val: u32, rotor_idx: usize, forward: bool) -> u32 {
        let step_val = self.step[rotor_idx];
        let effective_input = (val + step_val) % self.n;
        
        let wired_output = if forward {
            self.rotor_wirings[rotor_idx][effective_input as usize]
        } else {
            self.rotor_inv_wirings[rotor_idx][effective_input as usize]
        };
        
        (wired_output + self.n - step_val) % self.n 
    }
    
    pub fn call_char(&mut self, x: u32) -> u32 {
        // Sticking to stepping *after* encryption as per original code's comment.

        let mut current_val = x;

        // Forward path through rotors
        for i in 0..self.k {
            current_val = self.pass_rotor_through_idx(current_val, i, true);
        }

        // Reflector
        current_val = self.reflector_wiring[current_val as usize];

        // Backward path through rotors (in reverse order)
        for i in (0..self.k).rev() {
            current_val = self.pass_rotor_through_idx(current_val, i, false);
        }
        
        let encrypted_char = current_val;

        // Step for the next character (odometer style)
        if self.k > 0 { // Should always be true due to check in new()
            // Step the first rotor (fastest rotor, index 0)
            self.step[0] = (self.step[0] + 1) % self.n;
            
            // For subsequent rotors, step if the previous rotor just completed a full cycle
            for i in 0..(self.k - 1) { // Check rotors 0 to k-2 (i.e., previous rotors)
                if self.step[i] == 0 { // If rotor i (e.g., rotor 0) just turned over to 0
                    self.step[i+1] = (self.step[i+1] + 1) % self.n; // Step rotor i+1 (e.g., rotor 1)
                } else {
                    // If rotor i didn't turn over, then no subsequent rotors (i+1, i+2, ...) turn over
                    break; 
                }
            }
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
        self.step = vec![0; self.k]; // Reset all k steps to 0
    }
    
    /// Sets the rotor initial step positions
    pub fn set_steps(&mut self, steps: Vec<u32>) { // Argument type changed to Vec<u32>
        assert_eq!(steps.len(), self.k, "Number of steps provided must match number of rotors (k)");
        for i in 0..self.k {
            self.step[i] = steps[i] % self.n;
        }
    }

    /// Sets the rotor wirings and reflector.
    pub fn set_wirings(&mut self, rotor_wirings: Vec<Vec<u32>>, reflector_wiring: Vec<u32>) {
        assert_eq!(rotor_wirings.len(), self.k, "Number of rotor wirings provided must match k");
        for (i, wiring) in rotor_wirings.iter().enumerate() {
            assert_eq!(wiring.len() as u32, self.n, "Rotor {} wiring length mismatch", i);
            // TODO: Add validation for permutation property of each wiring
        }
        assert_eq!(reflector_wiring.len() as u32, self.n, "Reflector wiring length mismatch");
        // TODO: Add validation for reflector properties (self-inverse, no fixed points if n is even)

        self.rotor_inv_wirings = rotor_wirings.iter().map(|w| invert_permutation(w)).collect();
        self.rotor_wirings = rotor_wirings; 
        self.reflector_wiring = reflector_wiring;
    }

    #[cfg(test)]
    pub fn get_steps(&self) -> &Vec<u32> { // Return type changed
        &self.step
    }
    #[cfg(test)]
    pub fn get_rotor_wirings(&self) -> &Vec<Vec<u32>> { // Return type changed
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
        let perm = vec![2, 0, 1, 3];
        let inv = invert_permutation(&perm);
        assert_eq!(inv, vec![1, 2, 0, 3]);
    }

    #[test]
    fn test_reflector_generation_even() {
        let n = 4;
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let reflector = generate_reflector(n, &mut rng);
        for i in 0..n {
            assert_eq!(reflector[reflector[i as usize] as usize], i, "Not self-inverse at {}", i);
            assert_ne!(reflector[i as usize], i, "Fixed point at {}", i);
        }
    }
    
    #[test]
    fn test_reflector_generation_odd_exception() {
        let n = 3;
        let seed = [1u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let reflector = generate_reflector(n, &mut rng);
        let mut fixed_points = 0;
        for i in 0..n {
            assert_eq!(reflector[reflector[i as usize] as usize], i, "Not self-inverse at {}", i);
            if reflector[i as usize] == i {
                fixed_points += 1;
            }
        }
        assert_eq!(fixed_points, 1, "Expected one fixed point for odd n with this generator");
    }

    #[test]
    fn test_true_easy_enigma_encryption_self_reciprocal() {
        let n = 26; 
        let k = 2;  // Test with 2 rotors like original
        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(n, k, &mut rng);

        let plaintext_char = 5; 
        let ciphertext_char = enigma.call_char(plaintext_char);

        if n % 2 == 0 { // Reflector should not map a char to itself if n is even
            assert_ne!(plaintext_char, ciphertext_char, "A char should not encrypt to itself with a proper reflector for even n.");
        }

        enigma.reset_steps(); 
        let decrypted_char = enigma.call_char(ciphertext_char);

        assert_eq!(plaintext_char, decrypted_char, "Enigma should be self-reciprocal.");
    }

    #[test]
    fn test_true_easy_enigma_stepping_changes_output() {
        let n = 26;
        let k = 2; // Test with 2 rotors
        let seed = [1u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(n, k, &mut rng);
        
        let input_val = 10; 
        
        let first_encrypted = enigma.call_char(input_val);
        let second_encrypted = enigma.call_char(input_val); 
        
        assert_ne!(first_encrypted, second_encrypted, "Output should change due to rotor stepping.");
    }

    #[test]
    fn test_true_easy_enigma_reset_steps() {
        let n = 26;
        let k = 2; // Test with 2 rotors
        let seed = [2u8; 32];
        let mut rng = StdRng::from_seed(seed); // rng for initial enigma
        
        // Need a separate rng for the comparison enigma if we re-initialize with EasyEnigma::new
        // Or, better, clone the relevant parts.
        let mut enigma = EasyEnigma::new(n, k, &mut rng);
        
        // Save initial wirings for comparison later
        let initial_rotor_wirings = enigma.get_rotor_wirings().clone();
        let initial_reflector_wiring = enigma.get_reflector_wiring().clone();

        let input_array = vec![1, 2, 3, 4, 5];
        let _ = enigma.call(&input_array); 
        
        let steps_after_call = enigma.get_steps();
        assert_ne!(*steps_after_call, vec![0; k], "Steps should have advanced.");

        enigma.reset_steps();
        let steps_after_reset = enigma.get_steps();
        assert_eq!(*steps_after_reset, vec![0; k], "Steps should be reset to initial state.");
        
        let mut enigma_fresh_for_comparison = EasyEnigma {
            rotor_wirings: initial_rotor_wirings.clone(),
            rotor_inv_wirings: initial_rotor_wirings.iter().map(|w| invert_permutation(w)).collect(),
            reflector_wiring: initial_reflector_wiring.clone(),
            step: vec![0; k],
            n,
            k,
        };

        let first_run_output = enigma_fresh_for_comparison.call(&input_array);
        
        enigma.reset_steps(); 
        let second_run_output_after_reset = enigma.call(&input_array);

        assert_eq!(first_run_output, second_run_output_after_reset, "Encryption after reset should match initial encryption.");
    }

    #[test]
    fn test_enigma_k1_operation() {
        let n = 26;
        let k = 1;
        let seed = [3u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(n, k, &mut rng);

        let plaintext_char = 5; 
        // Initial state: step [0]
        let ciphertext_char1 = enigma.call_char(plaintext_char);
        // After call: step [1]
        assert_eq!(*enigma.get_steps(), vec![1]);

        let ciphertext_char2 = enigma.call_char(plaintext_char); 
        // After call: step [2]
        assert_eq!(*enigma.get_steps(), vec![2]);
        assert_ne!(ciphertext_char1, ciphertext_char2, "Output should change with stepping for k=1");

        // Test self-reciprocity for k=1
        enigma.set_steps(vec![0]); 
        assert_eq!(*enigma.get_steps(), vec![0]);
        
        let decrypted_char1 = enigma.call_char(ciphertext_char1);
        // After call: step is [1]
        assert_eq!(*enigma.get_steps(), vec![1]);
        assert_eq!(plaintext_char, decrypted_char1, "Self-reciprocity failed for k=1");
    }

    #[test]
    fn test_enigma_k3_stepping() {
        let n = 3; // Small n to make turnover happen quickly
        let k = 3;
        let seed = [4u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut enigma = EasyEnigma::new(n, k, &mut rng); // Initial steps [0,0,0]

        enigma.call_char(0); // Input char doesn't matter for testing steps
        assert_eq!(*enigma.get_steps(), vec![1,0,0], "Step after call 1");

        enigma.call_char(0);
        assert_eq!(*enigma.get_steps(), vec![2,0,0], "Step after call 2");

        enigma.call_char(0);
        assert_eq!(*enigma.get_steps(), vec![0,1,0], "Step after call 3 (rotor 0 turnover)");

        enigma.call_char(0);
        assert_eq!(*enigma.get_steps(), vec![1,1,0], "Step after call 4");
        enigma.call_char(0);
        assert_eq!(*enigma.get_steps(), vec![2,1,0], "Step after call 5");
        
        enigma.call_char(0);
        assert_eq!(*enigma.get_steps(), vec![0,2,0], "Step after call 6 (rotor 0 turnover)");

        enigma.call_char(0);
        assert_eq!(*enigma.get_steps(), vec![1,2,0], "Step after call 7");
        enigma.call_char(0);
        assert_eq!(*enigma.get_steps(), vec![2,2,0], "Step after call 8");
        
        enigma.call_char(0);
        assert_eq!(*enigma.get_steps(), vec![0,0,1], "Step after call 9 (rotor 0 and 1 turnover)");
    }
}