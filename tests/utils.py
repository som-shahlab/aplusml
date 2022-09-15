import sys
sys.path.append("..")
import sim

def check_history(simulation: sim.Simulation,
                  patient: sim.Patient,
                  current_state: str,
                  history: list):
    assert patient.current_state == current_state
    assert len(patient.history) == len(history)
    for idx in range(len(history)):
        assert patient.history[idx].state_id == history[idx]
        if idx > 0:
            state = simulation.states[patient.history[idx - 1].state_id]
            assert state.transitions[patient.history[idx - 1].transition_idx].dest == history[idx]

def check_probabilistic_outcomes_are_within_margin_of_error(counts: dict[int], probabilities: dict[float], margin_of_error: float):
    total_count = sum([val for val in counts.values()])
    for c in counts.keys():
        assert probabilities[c] - margin_of_error <= counts[c] / total_count <= probabilities[c] + margin_of_error