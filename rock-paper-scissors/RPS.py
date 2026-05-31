def player(prev_play, state={}):
    """Adaptive Rock/Paper/Scissors player for the freeCodeCamp project.

    The opponents in the challenge are deterministic, but use different rules.
    This player keeps a small set of opponent-model hypotheses and continuously
    scores which one best predicts the observed opponent moves. It then plays the
    move that beats the currently best prediction.
    """

    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    moves = ['R', 'P', 'S']

    if prev_play == '' or not state:
        state.clear()
        state.update({
            'opponent_history': [],
            'my_history': [],
            'turn': 0,
            'scores': {
                'quincy': 0,
                'kris': 0,
                'mrugesh': 0,
                'abbey': 0,
                'markov1': 0,
                'markov2': 0,
                'markov3': 0,
            },
            'last_predictions': {},
        })
    else:
        state['opponent_history'].append(prev_play)
        for name, prediction in state['last_predictions'].items():
            if prediction == prev_play:
                state['scores'][name] += 1

    opponent_history = state['opponent_history']
    my_history = state['my_history']
    turn = state['turn']

    def counter(move):
        return beats.get(move, 'R')

    def quincy_prediction():
        pattern = ['R', 'R', 'P', 'P', 'S']
        # quincy increments before playing, so turn 0 -> pattern[1].
        return pattern[(turn + 1) % len(pattern)]

    def kris_prediction():
        previous_my_play = my_history[-1] if my_history else 'R'
        return counter(previous_my_play)

    def mrugesh_prediction():
        last_ten = my_history[-10:]
        if not last_ten:
            most_frequent = 'S'
        else:
            most_frequent = max(moves, key=last_ten.count)
        return counter(most_frequent)

    def abbey_prediction():
        # Simulate Abbey's transition table from our move history. Abbey stores
        # our previous plays and predicts our next play from the last pair.
        play_order = {a + b: 0 for a in moves for b in moves}
        abbey_seen = []
        for my_play in my_history:
            abbey_seen.append(my_play)
            last_two = ''.join(abbey_seen[-2:])
            if len(last_two) == 2:
                play_order[last_two] += 1

        previous_my_play = my_history[-1] if my_history else 'R'
        potential_plays = [previous_my_play + move for move in moves]
        predicted_my_play = max(potential_plays, key=lambda pair: play_order[pair])[-1]
        return counter(predicted_my_play)

    def markov_prediction(order):
        if len(opponent_history) <= order:
            return quincy_prediction()

        pattern = ''.join(opponent_history[-order:])
        counts = {move: 0 for move in moves}
        for i in range(len(opponent_history) - order):
            if ''.join(opponent_history[i:i + order]) == pattern:
                counts[opponent_history[i + order]] += 1

        if max(counts.values()) == 0:
            return markov_prediction(order - 1) if order > 1 else quincy_prediction()
        return max(moves, key=lambda move: counts[move])

    predictions = {
        'quincy': quincy_prediction(),
        'kris': kris_prediction(),
        'mrugesh': mrugesh_prediction(),
        'abbey': abbey_prediction(),
        'markov1': markov_prediction(1),
        'markov2': markov_prediction(2),
        'markov3': markov_prediction(3),
    }

    # Give the hand-written opponent models a short warm-up, then trust the
    # model that has predicted the current match most accurately.
    if turn < 8:
        predicted_opponent_move = predictions['quincy']
    else:
        best_model = max(state['scores'], key=state['scores'].get)
        predicted_opponent_move = predictions[best_model]

    guess = counter(predicted_opponent_move)

    state['last_predictions'] = predictions
    state['my_history'].append(guess)
    state['turn'] += 1

    return guess
