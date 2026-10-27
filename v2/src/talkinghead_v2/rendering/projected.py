"""Bounded static canonical geometry for the native reference; no neural motion."""
from dataclasses import dataclass
import math
import struct

from talkinghead_v2.assets.avatar import MeshAsset
from talkinghead_v2.contracts import ContractError, Envelope, GenerationKey, MotionFrame, integer
from talkinghead_v2.rendering.mesh import deform


@dataclass(frozen=True)
class ProjectedMesh:
    resolution: int
    vertices: tuple
    faces: tuple

    def __post_init__(self):
        integer(self.resolution, 'native resolution', 16)
        if self.resolution > 512 or self.resolution % 2:
            raise ContractError('native static mesh requires an even resolution through 512')
        if type(self.vertices) is not tuple or not 3 <= len(self.vertices) <= 512 or type(self.faces) is not tuple or not 1 <= len(self.faces) <= 1024:
            raise ContractError('native mesh exceeds explicit geometry bounds; no truncation')
        for vertex in self.vertices:
            if (type(vertex) is not tuple or len(vertex) != 3 or
                any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1000000 for value in vertex) or vertex[2] <= 0):
                raise ContractError('native projected vertices need finite coordinates and positive camera depth')
        for face in self.faces:
            if (type(face) is not tuple or len(face) != 3 or len(set(face)) != 3 or
                any(type(index) is not int or not 0 <= index < len(self.vertices) for index in face)):
                raise ContractError('native face identity mismatch; no index clamping')

    def packed(self) -> bytes:
        xyz = tuple(value for vertex in self.vertices for value in vertex)
        indices = tuple(index for face in self.faces for index in face)
        return (b'THAMESH1' + struct.pack('<IIII', self.resolution, self.resolution, len(self.vertices), len(self.faces)) +
            struct.pack(f'<{len(xyz)}d', *xyz) + struct.pack(f'<{len(indices)}I', *indices))


def project_neutral(asset: MeshAsset, resolution=128) -> ProjectedMesh:
    """Preserve neutral rig/pose/camera semantics and exact face ordering."""
    integer(resolution, 'native resolution', 16)
    if resolution > 512 or resolution % 2:
        raise ContractError('native static mesh requires an even resolution through 512')
    key = GenerationKey('native-static-projection', 'neutral', 0)
    neutral = MotionFrame(Envelope(key, 0, 'native-static-projection'), asset.rig, 0, 48000, 48000,
        tuple(0. for _ in asset.rig.coefficient_names))
    vertices = deform(asset, neutral)
    projected = []
    for vertex in vertices:
        camera = tuple(sum(row[index] * vertex[index] for index in range(3)) + translation
            for row, translation in zip(asset.camera_rotation, asset.camera_translation, strict=True))
        projected.append(((camera[0] / asset.half_extent_m + 1) * (resolution - 1) / 2,
            (camera[1] / asset.half_extent_m + 1) * (resolution - 1) / 2, camera[2]))
    return ProjectedMesh(resolution, tuple(projected), asset.faces)
