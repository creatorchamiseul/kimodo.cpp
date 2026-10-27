#!/usr/bin/env python3
"""Build the somaskel77 expansion spec consumed by the BVH exporter and the
MotionForge Blender add-on.

Input: the Apache-2.0 skeleton assets from NVIDIA's reference implementation
(https://github.com/nv-tlabs/kimodo, kimodo/assets/skeletons/somaskel77/):

  joints.p                    77x3 neutral joint positions (torch zip pickle)
  relaxed_hands_rest_pose.npy 77x3x3 local rotation matrices, the relaxed-hand
                              rest pose the official 30->77 expansion starts from

Output: somaskel77_expanded.json - names, parents, parent-relative offsets and
relaxed rest rotations (XYZW quaternions) for all 77 joints. The 30 joints the
SOMA checkpoints actually predict are listed as "predicted"; at runtime their
rest rotations are overwritten with the model output, exactly like
SOMASkeleton30.to_SOMASkeleton77 upstream.

Also cross-checks, so a silently wrong conversion cannot be shipped:
  - the 30-joint somaskel30/joints.p must match the spec table the port's
    BVH exporter ships (soma30 cumulative offsets), within 1e-5;
  - the 77 joints.p shared joints must match the 30 joints.p, within 1e-5;
  - every rest rotation matrix must be orthonormal, within 1e-4.

Usage:
    python scripts/make_somaskel77_spec.py \
        --assets-dir assets/skeletons --out assets/skeletons/somaskel77
"""

import argparse
import io
import json
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np

NAMES30 = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]
PARENTS30 = [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13,
             3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28]

# The hierarchy exactly as SOMASkeleton77.bone_order_names_with_parents in
# nv-tlabs/kimodo kimodo/skeleton/definitions.py (Apache-2.0). Order matters:
# joints.p and relaxed_hands_rest_pose.npy are stored in this order.
NAMES77_WITH_PARENTS = [
    ("Hips", None),
    ("Spine1", "Hips"),
    ("Spine2", "Spine1"),
    ("Chest", "Spine2"),
    ("Neck1", "Chest"),
    ("Neck2", "Neck1"),
    ("Head", "Neck2"),
    ("HeadEnd", "Head"),
    ("Jaw", "Head"),
    ("LeftEye", "Head"),
    ("RightEye", "Head"),
    ("LeftShoulder", "Chest"),
    ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("LeftHandThumb1", "LeftHand"),
    ("LeftHandThumb2", "LeftHandThumb1"),
    ("LeftHandThumb3", "LeftHandThumb2"),
    ("LeftHandThumbEnd", "LeftHandThumb3"),
    ("LeftHandIndex1", "LeftHand"),
    ("LeftHandIndex2", "LeftHandIndex1"),
    ("LeftHandIndex3", "LeftHandIndex2"),
    ("LeftHandIndex4", "LeftHandIndex3"),
    ("LeftHandIndexEnd", "LeftHandIndex4"),
    ("LeftHandMiddle1", "LeftHand"),
    ("LeftHandMiddle2", "LeftHandMiddle1"),
    ("LeftHandMiddle3", "LeftHandMiddle2"),
    ("LeftHandMiddle4", "LeftHandMiddle3"),
    ("LeftHandMiddleEnd", "LeftHandMiddle4"),
    ("LeftHandRing1", "LeftHand"),
    ("LeftHandRing2", "LeftHandRing1"),
    ("LeftHandRing3", "LeftHandRing2"),
    ("LeftHandRing4", "LeftHandRing3"),
    ("LeftHandRingEnd", "LeftHandRing4"),
    ("LeftHandPinky1", "LeftHand"),
    ("LeftHandPinky2", "LeftHandPinky1"),
    ("LeftHandPinky3", "LeftHandPinky2"),
    ("LeftHandPinky4", "LeftHandPinky3"),
    ("LeftHandPinkyEnd", "LeftHandPinky4"),
    ("RightShoulder", "Chest"),
    ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("RightHandThumb1", "RightHand"),
    ("RightHandThumb2", "RightHandThumb1"),
    ("RightHandThumb3", "RightHandThumb2"),
    ("RightHandThumbEnd", "RightHandThumb3"),
    ("RightHandIndex1", "RightHand"),
    ("RightHandIndex2", "RightHandIndex1"),
    ("RightHandIndex3", "RightHandIndex2"),
    ("RightHandIndex4", "RightHandIndex3"),
    ("RightHandIndexEnd", "RightHandIndex4"),
    ("RightHandMiddle1", "RightHand"),
    ("RightHandMiddle2", "RightHandMiddle1"),
    ("RightHandMiddle3", "RightHandMiddle2"),
    ("RightHandMiddle4", "RightHandMiddle3"),
    ("RightHandMiddleEnd", "RightHandMiddle4"),
    ("RightHandRing1", "RightHand"),
    ("RightHandRing2", "RightHandRing1"),
    ("RightHandRing3", "RightHandRing2"),
    ("RightHandRing4", "RightHandRing3"),
    ("RightHandRingEnd", "RightHandRing4"),
    ("RightHandPinky1", "RightHand"),
    ("RightHandPinky2", "RightHandPinky1"),
    ("RightHandPinky3", "RightHandPinky2"),
    ("RightHandPinky4", "RightHandPinky3"),
    ("RightHandPinkyEnd", "RightHandPinky4"),
    ("LeftLeg", "Hips"),
    ("LeftShin", "LeftLeg"),
    ("LeftFoot", "LeftShin"),
    ("LeftToeBase", "LeftFoot"),
    ("LeftToeEnd", "LeftToeBase"),
    ("RightLeg", "Hips"),
    ("RightShin", "RightLeg"),
    ("RightFoot", "RightShin"),
    ("RightToeBase", "RightFoot"),
    ("RightToeEnd", "RightToeBase"),
]


def _float_storage(pid):
    return pid


_STORAGE_DTYPES = {
    "FloatStorage": "<f4", "DoubleStorage": "<f8", "HalfStorage": "<f2",
    "LongStorage": "<i8", "IntStorage": "<i4", "ShortStorage": "<i2",
    "CharStorage": "i1", "ByteStorage": "u1", "BoolStorage": "b1",
}


class _TorchStubUnpickler(pickle.Unpickler):
    """Read the data.pkl inside a torch.save zip without importing torch.

    torch serialises tensors as a pickle referencing raw storage buffers
    through persistent IDs, and rebuilds them with torch._utils helpers. The
    stubs below keep just enough of that to get the float data out.
    """

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        assert pid[0] == "storage", pid[0]
        dtype = _STORAGE_DTYPES[pid[1].__name__]
        return ("storage", pid[2], pid[4], dtype)

    def find_class(self, module, name):
        if module == "torch._utils" and name in (
                "_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {})
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        return super().find_class(module, name)


def _rebuild_tensor(storage, storage_offset, size, stride, *ignored):
    return ("tensor", storage, storage_offset, list(size), list(stride))


def _read_storage(zip_file, prefix, storage):
    _, key, numel, dtype = storage
    item = np.dtype(dtype).itemsize
    raw = zip_file.read("%sdata/%s" % (prefix, key))
    assert len(raw) >= numel * item, (key, len(raw), numel)
    return np.frombuffer(raw[:numel * item], dtype=dtype).copy()


def _read_tensor(zip_file, prefix, value):
    _, storage, offset, size, stride = value
    flat = _read_storage(zip_file, prefix, storage)
    if not size:
        return flat[0]
    if len(size) == 1:
        return flat[offset:offset + size[0]]
    # General small-tensor case: gather by stride.
    count = int(np.prod(size))
    index = np.zeros(count, dtype=np.int64)
    cursor = np.zeros(len(size), dtype=np.int64)
    for i in range(count):
        index[i] = offset + sum(cursor[d] * stride[d] for d in range(len(size)))
        for d in reversed(range(len(size))):
            cursor[d] += 1
            if cursor[d] < size[d]:
                break
            cursor[d] = 0
    return flat[index].reshape(size)


def load_torch_tensor(path):
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkl = [n for n in names if n.endswith("data.pkl")]
        assert len(pkl) == 1, names
        prefix = pkl[0][: -len("data.pkl")]
        unpickler = _TorchStubUnpickler(io.BytesIO(zf.read(pkl[0])))
        value = unpickler.load()
        if isinstance(value, tuple) and value and value[0] == "tensor":
            return _read_tensor(zf, prefix, value)
        # A bare storage was saved (rare) - return it raw.
        return _read_storage(zf, prefix, value[1], value[2])


def load_npy(path):
    with open(path, "rb") as handle:
        magic = handle.read(6)
        assert magic == b"\x93NUMPY", magic
        major, _minor = struct.unpack("<BB", handle.read(2))
        if major < 3:
            (header_len,) = struct.unpack("<H", handle.read(2))
        else:
            (header_len,) = struct.unpack("<I", handle.read(4))
        header = eval(handle.read(header_len).decode("latin1"))  # noqa: S307
        dtype = header["descr"]
        assert dtype == "<f4", dtype
        body = handle.read()
        return np.frombuffer(body, dtype="<f4").reshape(header["shape"])


def quat_xyzw_from_matrix(m):
    """Shepperd's method; w kept non-negative for a stable rest comparison."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = np.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = np.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = np.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    return [float(v / norm) for v in (x, y, z, w)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", default="assets/skeletons")
    parser.add_argument("--out", default="assets/skeletons/somaskel77")
    args = parser.parse_args()

    assets = Path(args.assets_dir)
    joints77 = load_torch_tensor(
        assets / "somaskel77" / "joints.p").reshape(77, 3)
    rest = load_npy(assets / "somaskel77" /
                    "relaxed_hands_rest_pose.npy").reshape(77, 3, 3)
    joints30 = load_torch_tensor(
        assets / "somaskel30" / "joints.p").reshape(30, 3)

    names77 = [n for n, _ in NAMES77_WITH_PARENTS]
    index77 = {n: i for i, n in enumerate(names77)}
    parents77 = [index77[p] if p is not None else -1
                 for _, p in NAMES77_WITH_PARENTS]

    # Upstream asserts the root sits at the origin; trust nothing.
    assert np.abs(joints77[0]).max() < 1e-6, joints77[0]

    # Check 1: the port's shipped soma30 table is the same skeleton upstream
    # trains on - otherwise the expansion would bind to a different body.
    neutral30 = [np.zeros(3)]
    for i in range(1, 30):
        neutral30.append(neutral30[PARENTS30[i]] + np.array(NAMES77_WITH_PARENTS[0] and _soma30_offset(i)))
    shipped = np.array(neutral30)
    diff30 = np.abs(shipped - joints30).max()
    assert diff30 < 1e-5, "soma30 table mismatch: %g" % diff30

    # Check 2: the 77 skeleton's shared joints sit exactly where the 30's do,
    # so expanding never disturbs the body motion.
    slice30_into_77 = [index77[n] for n in NAMES30]
    diff_shared = np.abs(joints77[slice30_into_77] - joints30).max()
    assert diff_shared < 1e-5, "shared joints mismatch: %g" % diff_shared

    # Check 3: rest rotations orthonormal.
    ident = np.eye(3)
    ortho = max(np.abs(np.einsum("fji,fjk->fik", rest, rest) - ident).max(),
                np.abs(np.linalg.det(rest) - 1.0).max())
    assert ortho < 1e-4, "rest rotations not orthonormal: %g" % ortho

    offsets = []
    for i in range(77):
        if parents77[i] < 0:
            offsets.append([0.0, 0.0, 0.0])
        else:
            offsets.append([float(v) for v in
                            joints77[i] - joints77[parents77[i]]])

    spec = {
        "format": "somaskel77-expansion-v1",
        "source": ("nv-tlabs/kimodo kimodo/assets/skeletons/somaskel77 "
                   "(Apache-2.0)"),
        "note": ("The 30 predicted joints overwrite their relaxed rest "
                 "rotations at runtime; the other 47 keep them."),
        "predicted_joints": NAMES30,
        "predicted_index_in_77": slice30_into_77,
        "names": names77,
        "parents": parents77,
        "offsets": offsets,
        "rest_rotations_xyzw": [quat_xyzw_from_matrix(m) for m in rest],
    }
    out = Path(args.out) / "somaskel77_expanded.json"
    out.write_text(json.dumps(spec, separators=(",", ":")) + "\n",
                   encoding="utf-8")
    print("wrote %s (%d bytes)" % (out, out.stat().st_size))
    print("soma30 table diff: %.2e | shared 77/30 diff: %.2e | ortho: %.2e"
          % (diff30, diff_shared, ortho))


def _soma30_offset(i):
    """The parent-relative offsets the port's BVH exporter ships."""
    fromsomabvh = [
        [0.0, 0.0, 0.0], [-0.00013727, 0.0500376256, -0.00053726669],
        [-1.86574103e-09, 0.0712530139, -0.000298248546],
        [-5.75188398e-09, 0.0755006305, -0.00815970992],
        [-0.00181676517, 0.263112953, -0.00553348292],
        [-2.85102231e-08, 0.0770939664, 0.0230258546],
        [-4.5975437e-08, 0.0612891595, 0.0195370861],
        [2.63687901e-05, 0.0047559225, 0.0309494062],
        [0.0320638079, 0.0538020513, 0.0758688308],
        [-0.0322244017, 0.05361869, 0.0755823359],
        [0.0162165175, 0.232371641, 0.0511341324],
        [0.149198457, 2.19397873e-08, -0.0550232576],
        [0.287393078, 2.50268389e-09, -2.58787737e-05],
        [0.270939812, -7.06625108e-09, 2.60897248e-05],
        [0.122686267, -0.0322017573, 0.0483306876],
        [0.190119595, -0.00312878387, -0.000339570373],
        [-0.0138011824, 0.231803086, 0.0521415786],
        [-0.150371962, 1.17387901e-07, -0.0554560437],
        [-0.287366393, 1.87628082e-08, -2.59709359e-05],
        [-0.271336198, -1.16767401e-09, 2.61269368e-05],
        [-0.122642483, -0.0321145448, 0.0480403904],
        [-0.190005945, -0.00306615542, -0.0003157343],
        [0.10043214, -0.0843452671, 0.0259565473],
        [-1e-08, -0.432217537, -0.00802912805],
        [1e-08, -0.421550959, -0.0348152298],
        [0.0, -0.0505947206, 0.132315294],
        [-0.10047278, -0.0829525995, 0.0262031695],
        [1e-08, -0.433622059, -0.00805555828],
        [2e-08, -0.421173943, -0.0347839785],
        [-3.42907669e-09, -0.0507960932, 0.132841956],
    ]
    return fromsomabvh[i]


if __name__ == "__main__":
    main()
