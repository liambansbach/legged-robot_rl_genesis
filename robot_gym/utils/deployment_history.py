"""Portable NumPy history and command-demand clock; no Genesis/RSL-RL imports.

Copy alongside an exported policy. Feed scaled features in contract order at each
genuine policy boundary. Sensors are already acquired/delayed upstream; this class
does not invent samples. Commands arriving between boundaries refresh live slots.
"""

import numpy as np


class PolicyHistory:
    def __init__(self, batch, length, features, live_indices):
        self.values = np.zeros((batch, length, features), dtype=np.float32)
        self.valid = np.zeros(batch, dtype=bool)
        self.live_indices = live_indices

    def reset(self, ids):
        self.valid[ids] = False

    def append(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        self.values[:, :-1] = self.values[:, 1:].copy()
        self.values[:, -1] = frame
        self.values[~self.valid] = frame[~self.valid, None]
        self.valid[:] = True
        return self.values

    def refresh(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        # After a same-boundary selective reset this is a first real measurement,
        # not another policy tick. Valid rows retain their complete sensor history.
        self.values[~self.valid] = frame[~self.valid, None]
        self.valid[:] = True
        self.values[:, -1, self.live_indices] = frame[:, self.live_indices]
        return self.values


def phase_features(phase, commands, axes, thresholds):
    """Smooth max-demand clock; axes and start/full thresholds come from the task."""
    limits = np.asarray(thresholds)
    demands = np.clip((abs(commands[:, axes]) - limits[:, 0]) / (limits[:, 1] - limits[:, 0]), 0, 1)
    demand = demands.max(axis=-1)
    demand = demand**2 * (3 - 2*demand)
    angle = 2*np.pi*phase
    return np.stack((np.sin(angle), np.cos(angle)), axis=-1) * demand[:, None]


def check_package(directory):
    """Replay the bundled real histories using only NumPy, PyTorch and ORT."""
    import json
    from pathlib import Path
    import torch
    import onnxruntime as ort
    path = Path(directory)
    contract = json.loads((path / 'contract.json').read_text(encoding='utf-8'))
    data = np.load(path / 'parity_samples.npz', allow_pickle=False)
    history = data['histories']
    model = torch.jit.load(str(path / 'estimated_policy.pt')).eval()
    session = ort.InferenceSession(str(path / 'policy.onnx'), providers=['CPUExecutionProvider'])
    maxima = [0., 0.]
    with torch.no_grad():
        for start in range(0, len(history), 256):
            frame = history[start:start+256]
            for i, (a, b) in enumerate(zip(model(torch.from_numpy(frame)), session.run(None, {'history': frame}))):
                np.testing.assert_allclose(a.numpy(), b, atol=2e-5, rtol=2e-5)
                maxima[i] = max(maxima[i], float(abs(a.numpy()-b).max()))
    reset = data['reset']
    time, batch = reset.shape
    sequence = history.reshape(time, batch, *history.shape[1:])
    state = PolicyHistory(batch, history.shape[1], history.shape[2],
                          contract['estimator']['features']['live_feature_indices'])
    history_error = 0.
    for t in range(time):
        state.reset(np.flatnonzero(reset[t]))
        rebuilt = state.append(sequence[t, :, -1])
        if reset[t].any():
            # Recorded first sensor acquisition, before same-boundary command refresh.
            state.values[reset[t]] = data['reset_frames'][t, reset[t], None]
            state.refresh(sequence[t, :, -1])
        np.testing.assert_allclose(rebuilt, sequence[t], atol=1e-6, rtol=0.)
        history_error = max(history_error, float(abs(rebuilt-sequence[t]).max()))
    field = next(f for f in contract['estimator']['features']['fields'] if f['name'] == 'previous_issued_clipped_action')
    result = dict(samples=len(history), action_max_error=maxima[0], velocity_max_error=maxima[1],
                  history_max_error=history_error, reset_samples=int(reset.sum()),
                  clipped_history_samples=int((abs(history[:, -1, field['start']:field['stop']]) >= contract['action_bound']-1e-6).any(1).sum()))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='Directory containing the interface-3 export')
    check_package(parser.parse_args().directory)
