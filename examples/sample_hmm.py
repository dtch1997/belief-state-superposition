from belief_state_superposition.hmm import sample_sequence

data = sample_sequence(16)
beliefs, states, emissions, next_beliefs, next_states = zip(*data)
print(beliefs)
print(states)
print(emissions)
