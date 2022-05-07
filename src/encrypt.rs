use rand::{thread_rng, ThreadRng}
use 

#[derive(Debug)]
struct EasyEncrypt {
    map: Vec<u32>,
    rng: ThreadRng,
}

impl EasyEncrypt {
    fn new(n: u32) {
        EasyEncrypt {
        map: mut (0..n).collect(),
        rng: mut thread_rng()
    }

    fn call(&mut self, x: u32) -> u32 {
        // useage vec.map(|x| encrypt.call(x));
        self.map.get(x).unwrap()
    }

    fn reset(&mut self) {
        self.rng.shuffle(&mut self.map);
    }
}