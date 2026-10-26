"""Offline canonical mesh import. No vendor loaders, pickle or topology repair."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from talkinghead_v2.assets.artifacts import sha256_file
from talkinghead_v2.contracts import ContractError, RigIdentity, integer, numeric_tuple


def digest(value) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()


def read_json(path: Path, limit=8*1024*1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ContractError("asset metadata must be a bounded regular file")
    return json.loads(path.read_text())


def points(value, name, count=None):
    if not isinstance(value, list) or (count is not None and len(value) != count):
        raise ContractError(f"{name} has a mismatched vertex count")
    result = tuple(tuple(row) if isinstance(row, list) else () for row in value)
    for row in result:
        numeric_tuple(row, name, 3)
    return result


@dataclass(frozen=True)
class MeshAsset:
    avatar_id: str
    tenant_id: str
    kind: str
    rig: RigIdentity
    vertices: tuple
    faces: tuple
    deltas: tuple
    bounds: tuple
    jaw_vertices: tuple
    jaw_pivot: tuple
    behavior_channels: tuple
    camera_rotation: tuple
    camera_translation: tuple
    half_extent_m: float
    manifest_sha256: str


def load_avatar(path: Path, *, tenant_id: str) -> MeshAsset:
    data = read_json(path)
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ContractError("unsupported avatar manifest")
    if data.get("tenant_id") != tenant_id or not isinstance(tenant_id, str) or not tenant_id:
        raise ContractError("avatar tenant mismatch")
    if data.get("status") != "approved":
        raise ContractError("avatar is not approved")
    kind = data.get("kind")
    required_use = "timing_fixture_only" if kind == "procedural_fixture" else "research_avatar"
    if kind not in {"procedural_fixture", "human_consented", "research_authorized"}:
        raise ContractError("avatar provenance kind is required")
    if not isinstance(data.get("allowed_uses"), list) or required_use not in data["allowed_uses"]:
        raise ContractError("avatar use is not authorized")
    provenance = data.get("provenance", {})
    for name in ("source", "geometry_terms", "texture_terms", "authorization_record_id"):
        if not isinstance(provenance.get(name), str) or not provenance[name].strip():
            raise ContractError("explicit source, terms and authorization record are required")
    if kind == "human_consented" and not provenance.get("consent_record_id"):
        raise ContractError("human avatar needs a consent record")
    appearance = data.get("appearance", {})
    if appearance.get("representation") != "mesh_json":
        raise ContractError("only mesh_json is implemented; Gaussian integration remains unqualified")
    name = appearance.get("path")
    if not isinstance(name, str) or PurePosixPath(name).name != name or name in ("", ".", "..") or "\\" in name:
        raise ContractError("mesh must be a local flat asset filename")
    mesh_path = path.parent / name
    mesh = read_json(mesh_path)
    if "sha256:" + sha256_file(mesh_path) != appearance.get("sha256"):
        raise ContractError("mesh file hash mismatch")
    if not isinstance(mesh, dict):
        raise ContractError("mesh must be a JSON object")
    vertices = points(mesh.get("vertices"), "vertices")
    if not 3 <= len(vertices) <= 20000:
        raise ContractError("mesh vertex count is outside the CPU fallback bound")
    faces = mesh.get("faces")
    if not isinstance(faces, list) or not 1 <= len(faces) <= 40000:
        raise ContractError("mesh needs bounded triangular faces")
    for face in faces:
        if (not isinstance(face, list) or len(face) != 3 or len(set(face)) != 3 or
            any(type(i) is not int or not 0 <= i < len(vertices) for i in face)):
            raise ContractError("invalid face index; clamping/truncation is forbidden")
    raw_rig = data.get("rig", {})
    if raw_rig.get("units") != "meters" or raw_rig.get("rotation") != "quaternion_xyzw":
        raise ContractError("unsupported rig units or rotation convention")
    names = raw_rig.get("coefficient_names", [])
    rig = RigIdentity(raw_rig.get("schema_id"), raw_rig.get("basis_hash"), raw_rig.get("topology_hash"),
        tuple(names), raw_rig.get("coordinates"), raw_rig.get("pose_semantics"))
    basis = mesh.get("blendshapes")
    if not isinstance(basis, dict) or set(basis) != set(names):
        raise ContractError("blendshape names must match the declared rig exactly")
    authority = data.get("channel_authority",{})
    behavior, articulation = authority.get("behavior"),authority.get("articulation")
    if (not isinstance(behavior,list) or not isinstance(articulation,list) or
        any(not isinstance(name,str) for name in behavior+articulation) or
        len(set(behavior+articulation)) != len(behavior+articulation) or
        set(behavior+articulation) != set(names)):
        raise ContractError("channel authority must explicitly partition behavior and articulation")
    deltas = tuple(points(basis[name], name, len(vertices)) for name in names)
    if digest({"vertices": mesh["vertices"], "faces": faces}) != rig.topology_hash:
        raise ContractError("topology/neutral geometry hash mismatch")
    basis_identity = {k:mesh.get(k) for k in ("blendshapes", "jaw_vertices", "jaw_pivot_m")}
    if digest(basis_identity) != rig.basis_hash:
        raise ContractError("basis/jaw binding hash mismatch")
    bounds = mesh.get("bounds", {})
    if not isinstance(bounds, dict) or set(bounds) != set(names):
        raise ContractError("every coefficient needs calibrated bounds")
    limits = tuple(tuple(bounds[name]) for name in names)
    for limit in limits:
        numeric_tuple(limit, "coefficient bounds", 2)
        if not limit[0] <= 0 <= limit[1] or limit[0] == limit[1]:
            raise ContractError("coefficient bounds must contain neutral zero")
    jaw = mesh.get("jaw_vertices")
    if (not isinstance(jaw, list) or len(jaw) != len(set(jaw)) or
        any(type(i) is not int or not 0 <= i < len(vertices) for i in jaw)):
        raise ContractError("invalid jaw vertex binding")
    pivot = tuple(mesh.get("jaw_pivot_m", []))
    numeric_tuple(pivot, "jaw pivot", 3)
    camera = data.get("camera", {})
    if camera.get("projection") != "orthographic" or camera.get("coordinates") != "opencv_x_right_y_down_z_forward":
        raise ContractError("unsupported explicit camera convention")
    rotation = points(camera.get("world_to_camera_rotation"), "camera rotation", 3)
    translation = tuple(camera.get("world_to_camera_translation_m", []))
    numeric_tuple(translation, "camera translation", 3)
    for i in range(3):
        for j in range(3):
            if not math.isclose(sum(rotation[i][k]*rotation[j][k] for k in range(3)), float(i==j), abs_tol=1e-6):
                raise ContractError("camera rotation must be orthonormal")
    a,b,c = rotation
    det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if not math.isclose(det, 1, abs_tol=1e-6):
        raise ContractError("camera reflection is not a rotation")
    extent = camera.get("half_extent_m")
    if type(extent) not in (float,int) or not math.isfinite(extent) or not 0 < extent <= 10:
        raise ContractError("invalid orthographic camera extent")
    avatar_id = data.get("avatar_id")
    if not isinstance(avatar_id, str) or not avatar_id.strip():
        raise ContractError("avatar ID is required")
    return MeshAsset(avatar_id, tenant_id, kind, rig, vertices, tuple(tuple(f) for f in faces), deltas,
        limits, tuple(jaw), pivot, tuple(behavior), rotation, translation, extent, sha256_file(path))
