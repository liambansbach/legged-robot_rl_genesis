"""Verify this copied package without a simulator or training repository."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys
import xml.etree.ElementTree as ET


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_file(root, name):
    if Path(name).is_absolute() or PureWindowsPath(name).drive or '://' in name:
        raise ValueError(f'Expected a relative package path: {name}')
    path = (root / name).resolve()
    path.relative_to(root)
    # Windows also checks spelling/case, so a Linux copy has the same closure.
    current = root
    for part in path.relative_to(root).parts:
        if part not in {p.name for p in current.iterdir()}:
            raise ValueError(f'Missing or incorrectly cased dependency: {name}')
        current = current / part
    if not path.is_file():
        raise ValueError(f'Missing file: {name}')
    return path


def check_files(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    files = {p.relative_to(root).as_posix() for p in root.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    if files != set(manifest['files']):
        raise ValueError('Package files differ from manifest')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Package must not contain symlinks')
    for name, record in manifest['files'].items():
        path = relative_file(root, name)
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'File identity mismatch: {name}')
    pending, seen = [manifest['robot']['urdf']], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = relative_file(root, name)
        seen.add(name)
        references = []
        if path.suffix == '.urdf':
            references = [x.get('filename') for x in ET.parse(path).findall('.//mesh')]
            references += [x.get('filename') for x in ET.parse(path).findall('.//texture')]
        elif path.suffix.lower() == '.dae':
            references = [x.text for x in ET.parse(path).findall('.//{*}library_images/{*}image/{*}init_from')]
        for ref in references:
            if PureWindowsPath(ref).drive or Path(ref).is_absolute() or '://' in ref:
                raise ValueError(f'Nonportable asset reference: {ref}')
            target = (path.parent / ref).resolve().relative_to(root).as_posix()
            relative_file(root, target)
            pending.append(target)
    if seen != set(manifest['robot']['dependency_closure']):
        raise ValueError('Robot dependency closure differs from manifest')
    contract = json.loads((root / 'contract.json').read_text())
    if contract['interface_version'] != 3 or contract['external_input']['shape'] != ['batch', 5, 55]:
        raise ValueError('Expected the frozen history interface')
    if (contract['action_dim'] != 16 or contract['actor_input_dim'] != 58
            or contract['checkpoint_sha256'] != manifest['selected']['checkpoint_sha256']):
        raise ValueError('Candidate/interface mismatch')
    return manifest, contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torchscript', action='store_true', help='Also check both TorchScript files using PyTorch')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest, contract = check_files(root)
    sys.path.insert(0, str(root))  # Only this portable directory, also under python -I.
    import numpy as np
    import onnxruntime as ort
    from deployment_history import check_package, phase_features
    graph = ort.InferenceSession(str(root / 'policy.onnx'), providers=['CPUExecutionProvider'])
    inputs, outputs = graph.get_inputs(), graph.get_outputs()
    assert [(x.name, x.shape, x.type) for x in inputs] == [('history', ['batch', 5, 55], 'tensor(float)')]
    assert [(x.name, x.shape, x.type) for x in outputs] == [
        ('clipped_actions', ['batch', 16], 'tensor(float)'), ('velocity_m_s', ['batch', 3], 'tensor(float)')]
    for filename in ('parity_samples.npz', 'validation/lifecycle_samples.npz'):
        with np.load(root / filename) as data:
            for field in ('histories', 'expected_actions', 'expected_velocity'):
                if data[field].dtype != np.float32 or not np.isfinite(data[field]).all():
                    raise ValueError(f'Invalid float32 fixture: {filename}/{field}')
        print(filename)
        check_package(root, torchscript=args.torchscript, samples_file=filename)
    # Native lifecycle smoke: explicit reset settling gives .025; selective reset
    # at tick75 starts row0 at zero without adding a settling step to row1.
    with np.load(root / 'validation/lifecycle_samples.npz') as data:
        sequence = data['histories'].reshape(150, 2, 5, 55)
        p = np.full(2, .025, dtype=np.float32)
        for tick in range(150):
            if tick == 75:
                p[0] = 0.
            expected = phase_features(p, sequence[tick, :, -1, 6:9], [1, 2], [[.01, .05], [.10, .25]])
            np.testing.assert_allclose(expected, sequence[tick, :, -1, 53:55], atol=2e-6, rtol=0)
            p = (p + np.float32(contract['policy_dt'] / contract['phase']['period_s'])) % 1
    if args.torchscript:
        import torch
        reference = np.load(root / 'reference/parity_samples.npz')
        actor = torch.jit.load(str(root / 'reference/policy.pt'), map_location='cpu').eval()
        with torch.no_grad():
            actual = actor(torch.from_numpy(reference['observations'])).numpy()
        np.testing.assert_allclose(actual, reference['expected_clipped_actions'], atol=2e-6, rtol=0)
        np.testing.assert_allclose(actual, np.clip(reference['recorded_gpu_raw_actions'], -1, 1), atol=5e-5, rtol=0)
    assert not any(name in sys.modules for name in ('genesis', 'rsl_rl', 'robot_gym'))
    print(f"PASS: {len(manifest['files'])} file hashes; {len(manifest['robot']['dependency_closure'])} robot assets; history, phase and inference parity.")
    print('Development baseline for diagnostic Sim2Sim; stopping limitations unresolved; not hardware-qualified.')


if __name__ == '__main__':
    main()
