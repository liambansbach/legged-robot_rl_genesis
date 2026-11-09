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


def check_package(directory, torchscript=True, samples_file='parity_samples.npz'):
    """Check ORT against saved eager outputs; optionally also check TorchScript."""
    import json
    from pathlib import Path
    import onnxruntime as ort
    path = Path(directory)
    contract = json.loads((path / 'contract.json').read_text(encoding='utf-8'))
    data = np.load(path / samples_file, allow_pickle=False)
    history = data['histories']
    model = None
    if torchscript:
        import torch
        model = torch.jit.load(str(path / 'estimated_policy.pt')).eval()
    session = ort.InferenceSession(str(path / 'policy.onnx'), providers=['CPUExecutionProvider'])
    maxima = [0., 0.]
    for start in range(0, len(history), 256):
        frame = history[start:start+256]
        actual = session.run(None, {'history': frame})
        if 'expected_actions' in data:
            expected = [data[k][start:start+256] for k in ('expected_actions', 'expected_velocity')]
        elif model is not None:  # Earlier interface-3 exports did not save eager outputs.
            with torch.no_grad():
                expected = [x.numpy() for x in model(torch.from_numpy(frame))]
        else:
            raise ValueError('This older export needs PyTorch or a fresh export with reference outputs')
        for i, (a, b) in enumerate(zip(expected, actual)):
            np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
            maxima[i] = max(maxima[i], float(abs(a-b).max()))
        if model is not None:
            with torch.no_grad():
                for a, b in zip(expected, model(torch.from_numpy(frame))):
                    np.testing.assert_allclose(a, b.numpy(), atol=2e-5, rtol=2e-5)
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
    result = dict(samples=len(history), action_max_error=maxima[0], velocity_max_error=maxima[1],
                  history_max_error=history_error, reset_samples=int(reset.sum()))
    for field in contract['estimator']['features']['fields']:
        if field['name'] == 'previous_issued_clipped_action':
            actions = history[:, -1, field['start']:field['stop']]
            result['clipped_history_samples'] = int((abs(actions) >= contract['action_bound']-1e-6).any(1).sum())
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='Directory containing the interface-3 export')
    parser.add_argument('--no-torch', action='store_true', help='Check ORT against saved eager outputs only')
    args = parser.parse_args()
    check_package(args.directory, torchscript=not args.no_torch)
