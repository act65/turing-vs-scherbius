use rand::{
    Rng,
    rngs::ThreadRng
};

type Cards = Vec<u32>;

pub fn sample_random_ints(n: u32, max: u32, rng: &mut ThreadRng) -> Vec<u32> {
    (0..n).map(|_| rng.gen_range(1..max)).collect()
}

fn flatten<T>(v: &Vec<Vec<T>>) -> Vec<T>
where
    T: Copy, // Ensure T implements Copy, so we can dereference safely
{
    let mut result: Vec<T> = Vec::new();
    for x in v {
        for y in x {
            result.push(*y);
        }
    }
    result
}

pub fn remove_played_cards_from_hand(hand: &mut Cards, played: &Vec<Cards>) {
    let flat_played: Vec<u32> = flatten(played);
    for c in flat_played.iter() {
        if let Some(index) = hand.iter().position(|&y| y == *c) {
            hand.remove(index);
        }
    }
}

// tests

#[cfg(test)]
// test random ints
mod test_random_ints {

    #[test]
    fn test_sample_random_ints() {
        let mut rng = thread_rng();
        let n = 10;
        let max = 100;
        let result = sample_random_ints(n, max, &mut rng);
        assert_eq!(result.len(), n as usize);
        for x in result {
            assert!(x < max);
        }
    }
}
// test flatten
mod test_flatten {

    #[test]
    fn test_flatten() {
        let v = vec![vec![1, 2], vec![3, 4]];
        let result = flatten(&v);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }
}