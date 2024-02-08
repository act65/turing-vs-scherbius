## Scherbius versus Turing

__Arthur Scherbius__ built the Engima encryption machine. It was used in World War 2 to encrypt German miliarly communications. __Alan Turing__ is credited with 'breaking' the Enigma encrpytion and using the intel to win the war. 

> Alan Turing: We need your help, to keep this a secret from Admiralty, Army, RAF, uh...as no one can know, that we've broken Enigma, not even Dennison.<br>
...<br>
Alan Turing: While we develop a system to help you determine how much intelligence to act on. Which, uh, attacks to stop, which to let through. Statistical analysis, the minimum number of actions it will take, for us to win the war - but the maximum number we can take before the Germans get suspicious.
(from The Imitation Game)

***

To win this game as 'Turing', this game requires you to;
- A) break a code,
- B) exploit the broken code, without revealing you have broken it. 

***

The idea for this game is from a friend (Nick Johnstone aka Widdershin).
https://replit.com/@Widdershin/TuringVsScherbius#main.rb

I have implemented his idea with some small changes;

- encryption is based on a 'simple' version of enimga. Aka, a 2 rotor polynumeric substitution cipher.
- re-encryption now costs victory points
- Can send as many or as few resources to a single battle as you like (rather than max 2).
- The code supports any number of battles.

***

Properties.
- Asymmetric.
- Partial info.
- Discrete states and actions.

## Installation / playing

(assumes you have rust installed)

```bash
git clone git@github.com:act65/turing-vs-scherbius.git
cd turing-vs-scherbius
cargo run
```

## Development

- python api (WIP using pyo3)
- add 'map' / topology to battles. they have locations. resources (such as food / ammo / weapons / soldiers ...) must be moved between them (rather than just allocated).
- implement some decent computer players (via RL)
- hyper search through different configs to find the 'best' verson of this game (most even, many counter plays, )
- use a seed for the random number generators
