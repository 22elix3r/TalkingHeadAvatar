from dataclasses import replace
from pathlib import Path
import struct

import pytest

from talkinghead_v2.assets.avatar import load_avatar
from talkinghead_v2.contracts import ContractError
from talkinghead_v2.rendering.projected import ProjectedMesh, project_neutral


def test_neutral_projection_preserves_camera_depth_topology_and_wire_values():
    asset = load_avatar(Path(__file__).parents[1] / 'src/talkinghead_v2/fixtures/p03-avatar/avatar.json', tenant_id='procedural')
    mesh = project_neutral(asset)
    assert mesh.faces is asset.faces
    assert len(mesh.vertices) == len(asset.vertices)
    first = asset.vertices[0]
    camera = [sum(row[i] * first[i] for i in range(3)) + translation
        for row, translation in zip(asset.camera_rotation, asset.camera_translation)]
    assert mesh.vertices[0] == ((camera[0] / asset.half_extent_m + 1) * 63.5,
        (camera[1] / asset.half_extent_m + 1) * 63.5, camera[2])
    data = mesh.packed()
    assert data[:8] == b'THAMESH1'
    assert struct.unpack_from('<IIII', data, 8) == (128, 128, len(asset.vertices), len(asset.faces))
    assert struct.unpack_from('<3d', data, 24) == mesh.vertices[0]
    assert len(data) == 24 + len(asset.vertices) * 24 + len(asset.faces) * 12
    assert data == project_neutral(asset).packed()


@pytest.mark.parametrize('vertices,faces', [
    (((0., 0., 0.), (1., 0., 1.), (0., 1., 1.)), ((0, 1, 2),)),
    (((0., 0., float('nan')), (1., 0., 1.), (0., 1., 1.)), ((0, 1, 2),)),
    (((0., 0., 1.), (1., 0., 1.), (0., 1., 1.)), ((0, 1, 3),)),
    (((0., 0., 1.), (1., 0., 1.), (0., 1., 1.)), ((0, 0, 2),)),
    (((0., 0., 1.), (1., 0., 1.), (0., 1., 1.)), ((False, 1, 2),)),
])
def test_native_projection_rejects_invalid_binding_instead_of_repair(vertices, faces):
    with pytest.raises(ContractError): ProjectedMesh(128, vertices, faces)


def test_projection_rejects_unsupported_shape_and_camera_near_plane():
    vertices = ((0., 0., 1.), (1., 0., 1.), (0., 1., 1.))
    for resolution in [15, 127, 514]:
        with pytest.raises(ContractError): ProjectedMesh(resolution, vertices, ((0, 1, 2),))
    with pytest.raises(ContractError): ProjectedMesh(128, vertices * 200, ((0, 1, 2),))
    asset = load_avatar(Path(__file__).parents[1] / 'src/talkinghead_v2/fixtures/p03-avatar/avatar.json', tenant_id='procedural')
    for resolution in ['128', 127, 514]:
        with pytest.raises(ContractError): project_neutral(asset, resolution)
    with pytest.raises(ContractError): project_neutral(replace(asset, camera_translation=(0., 0., -10.)))
