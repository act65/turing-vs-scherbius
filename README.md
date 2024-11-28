# Scherbius versus Turing

__Arthur Scherbius__ built the Engima encryption machine. It was used in World War 2 to encrypt German military communications. __Alan Turing__ is credited with 'breaking' the Enigma encrpytion and using the intel to win the war. 

> Alan Turing: We need your help, to keep this a secret from Admiralty, Army, RAF, uh...as no one can know, that we've broken Enigma, not even Dennison.<br>
...<br>
Alan Turing: While we develop a system to help you determine how much intelligence to act on. Which, uh, attacks to stop, which to let through. Statistical analysis, the minimum number of actions it will take, for us to win the war - but the maximum number we can take before the Germans get suspicious.
(quote from The Imitation Game)

To win this game as 'Turing', this game requires you to;
- A) break a code,
- B) exploit the broken code, without revealing you have broken it. 

# Overview

The game is a turn-based, two player game.
One player plays as 'Turing' and the other as 'Scherbius'.

It has 1 number of hidden information.
The game is played with cards.

## Game logic

The idea for this game is from a friend (Nick Johnstone aka Widdershin).
https://replit.com/@Widdershin/TuringVsScherbius#main.rb

I have implemented the core game in rust with some small changes;

- encryption is based on a 'simple' version of enimga. Aka, a 2 rotor polynumeric substitution cipher.
- re-encryption now costs victory points
- you can send as many or as few resources to a single battle as you like (rather than max 2).

See 'src/' for the rust code and game logic.

## RL agent

I have implemented a RL agent to play the game. 


This is in the 'rl/' directory.


## Pygame interface

I have also implemented a pygame interface for the game which supports the following features;

- using custom images for the game pieces.
- 

See 'py/' for the game interface.

# Installation

First, clone the repository;

```bash
git clone ...
cd turing-vs-scherbius
```

The source for the game is written in rust. To build the game, you will need to have rust and pyo3 installed.

To install the game, run the following command;

```bash
pip install pyo3
maturin init
maturin develop
```