use rand::{
    thread_rng,
    Rng,
    rngs::ThreadRng
};

#[derive(Debug)]
pub struct EasyEnigma {
    // just 2 rotors
    rotor: [u32; 2],
    step: [u32; 2],
    n: u32,
}

impl EasyEnigma {
    pub fn new(n: u32) -> EasyEnigma {
        let mut rng = thread_rng();
        
        EasyEnigma {
            rotor: [rng.gen_range(1..n), rng.gen_range(1..n)],
            step: [0, 0],
            n: n,
        }
    }

    pub fn call(&mut self, array: &Vec<u32>) -> Vec<u32> {
        let mut encrypted_array: Vec<u32> = Vec::new();

        for x in array {
                let y = (*x % (self.rotor[0]+self.step[0]+1)) % (self.rotor[1]+self.step[1]+1);
                encrypted_array.push(y);

                self.step[0] = (self.step[0]+1) % self.n;
                if self.step[0] % self.n == 0 {
                    self.step[1] = (self.step[1]+1) % self.n;
                }
            }

            return encrypted_array
    }

    pub fn reset(&mut self) {
        self.step = [0, 0];
    }

    pub fn set(&mut self, x: [u32; 2]) {
        self.rotor = x
    }
}
