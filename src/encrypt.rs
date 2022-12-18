use rand::{thread_rng, ThreadRng} 

#[derive(Debug)]
struct EasyEncrypt {
    map: Vec<u32>,
    rng: ThreadRng,
}

impl EasyEncrypt {
    fn new(n: u32) {
        let map = mut (0..n).collect();
        let rng = mut thread_rng();
        
        rng.shuffle(&mut map)

        EasyEncrypt {
        map: map,
        rng: rng
        }
    }

    fn call(&mut self, array: Vec<u32>) -> Vec<u32> {
        let mut encrypted_array = Vec<u32>::new();
        for x in array {
                let y = self.map.get(x).unwrap();
                encrypted_array.push(y)
            }
        return encrypted_array
    }

    fn reset(&mut self) {
        self.rng.shuffle(&mut self.map);
    }
}