class Player():
    def __init__(self, config):
        self.config = config

    def choose_action(self, game_state):
        raise NotImplementedError

    def __call__(self, game_state):
        raise NotImplementedError

class Scherbius(Player):
    def __call__(self, game_state):
        del game_state.turing_hand

class RandomScherbius(Scherbius):
    def choose_action(hand, num_battles):
        strategy = [[] for _ in range(num_battles)]
        hand_copy = list(hand)
        random.shuffle(hand_copy)
        for i in range(num_battles):
            if not hand_copy: break
            if random.random() < 0.9:
                num_cards_to_play = random.randint(1, min(GAME_CONFIG.max_cards_per_battle, len(hand_copy)))
                for _ in range(num_cards_to_play):
                    if hand_copy:
                        strategy[i].append(hand_copy.pop())
        encrypt = random.choice([True, False])
        return strategy, encrypt

def turing_ai_player(hand, num_battles):
    strategy = [[] for _ in range(num_battles)]
    hand_copy = list(hand)
    random.shuffle(hand_copy)
    for i in range(num_battles):
        if not hand_copy: break
        if random.random() < 0.7:
            num_cards_to_play = random.randint(1, min(GAME_CONFIG.max_cards_per_battle, len(hand_copy)))
            for _ in range(num_cards_to_play):
                if hand_copy:
                    strategy[i].append(hand_copy.pop())
    return strategy
