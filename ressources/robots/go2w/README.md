# Go2-W robot assets

`urdf/go2w_measured_ed8dc93.urdf` is the current model. Its exact bytes are
SHA256 `d298cc7bf4894e869840bdab9ac60f09548d998444018e61854636cff46b1d8c`.
The copy in `ressources/pretrained/go2w/go2w_locomotion/robot/` is a portable snapshot, checked
against this source; do not edit the snapshot independently.

The other retained file, `go2w_reference_original.urdf`, is a historical thesis
model, **not a pristine manufacturer export**. It matches this repository at
`0ca8d747a94341ed94d59075c2899c5348de688a` after CRLF-to-LF normalization.
It already has cylindrical wheel collisions and the thesis camera assembly.
The snapshots were recorded in `372dd1de5cc8f4cb135b3f475f96de1a38384d58`;
the measured model came from navigation commit
`ed8dc93b3065a8e2a3a5919f032ed4690529117c`. See the unchanged historical
[native reference](../../../docs/go2w_native_reference.json) for that provenance.
Both the current composed policy and its privileged parent actor use the
**measured** model. The historical reference URDF is for older comparisons only.

| XML reviewed | Total authored mass | Wheel cylinder radius | Camera assembly |
| --- | ---: | ---: | --- |
| Retained reference | 19.523 kg | .086 m | Geometry present, camera/mount inertials zero |
| Removed `go2w_description.urdf` | 19.63898 kg | .086 m | Intermediate .11598 kg camera body with estimated COM/inertia |
| Current measured model | 19.68371 kg | .09167 m | .12072 kg camera plus .03999 kg mounts/adapter; estimated COM/inertias |

All three have 40 authored links, 39 joints (16 actuated), identical actuator
order, joint axes and limits. Order is FL, FR, RL, RR, each hip/thigh/calf/wheel;
hip axes are +X, other actuated axes +Y. Wheel collision offsets and .052 m
width are unchanged. The measured radius comes from the owner's approximate
.576 m loaded tread circumference, **not an identified rolling radius**. Wheel
visual X/Z scaling changes; vendor wheel inertias remain unchanged.

Relative to the reference, the measured camera base moves from
`(.295, 0, .075)` to `(.29236, 0, .079)` m. Other fixed transforms, including
the authored `imu` mount `(-.02557, 0, .04232)` m with zero RPY, are unchanged.
Only camera/mount masses, COMs and inertias, that camera base transform, wheel
radius and visual scaling differ. The URDF comments distinguish owner mass/pose
measurements from uniform-density STL COM/inertia estimates and approximate
mount/adapter assumptions. Those comments and all measured bytes are preserved.

The removed intermediate file was the camera-body update in
`c2c2febed2fc4aea20d874876091a3377f2e9ed2`; it was neither the selected measured
model nor the original published policy's model. Current visualization launch
files now explicitly use the measured model. No old filename is redirected to a
different robot. Recover incompatible older policies **with their matching code
and assets** at `7d1f87c87fe1b190b4278fe870c69d6c652a3ce5` or their recorded
historical revision. The diagnostic geometry test retains its reference model.

Both retained models use the same 13 mesh files in `dae/`; none was removed.
Case-sensitive names, relative references and authored attribution are retained.
The current portable policy includes that complete dependency closure.
