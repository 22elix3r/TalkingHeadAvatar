"""Explicit, bounded calibration. Equal coefficient counts never imply compatibility."""

from dataclasses import dataclass, replace
import math
import re

from talkinghead_v2.assets.avatar import MeshAsset, points
from talkinghead_v2.contracts import ContractError, MotionFrame, RigIdentity, numeric_tuple


@dataclass(frozen=True)
class RetargetProfile:
    source: RigIdentity
    target: RigIdentity
    matrix: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    calibration_revision: str

    def __post_init__(self):
        if self.source.coordinates != self.target.coordinates or self.source.pose_semantics != self.target.pose_semantics:
            raise ContractError("retarget transforms need matching explicit coordinate/pose conventions")
        if not isinstance(self.calibration_revision,str) or not re.fullmatch(r"sha256:[0-9a-f]{64}",self.calibration_revision):
            raise ContractError("calibration revision must be a content hash")
        if not isinstance(self.matrix, tuple) or len(self.matrix) != len(self.target.coefficient_names):
            raise ContractError("retarget matrix needs one row per target coefficient")
        for row in self.matrix:
            numeric_tuple(row, "retarget row",len(self.source.coefficient_names))
        numeric_tuple(self.bias,"retarget bias",len(self.matrix))
        if not isinstance(self.bounds,tuple) or len(self.bounds) != len(self.matrix):
            raise ContractError("retarget bounds mismatch")
        for bound,bias in zip(self.bounds,self.bias,strict=True):
            numeric_tuple(bound,"retarget bounds",2)
            if bound[0] >= bound[1] or not bound[0] <= bias <= bound[1]:
                raise ContractError("neutral retarget bias is outside calibrated bounds")

    def apply(self, frame: MotionFrame) -> MotionFrame:
        if frame.rig != self.source:
            raise ContractError("source basis/topology/order does not match calibration")
        values = tuple(max(low,min(high,bias+sum(w*x for w,x in zip(row,frame.coefficients,strict=True))))
            for row,bias,(low,high) in zip(self.matrix,self.bias,self.bounds,strict=True))
        # Jaw/head remain rigid transforms, never hidden blendshape slots.
        return replace(frame,rig=self.target,coefficients=values)


def fit_landmarks(asset: MeshAsset, indices: tuple[int, ...], target: tuple, *, regularization=1e-8,
                  iterations=200) -> dict:
    """Bounded coordinate-descent fit of explicit corresponding target landmarks.

    Caller must supply correspondences and a separately jaw-aligned target. This
    solves mesh coefficients; it does not invent a Claire-to-FLAME mapping.
    """
    if not isinstance(indices,tuple) or not indices or len(set(indices)) != len(indices) or any(
        type(i) is not int or not 0 <= i < len(asset.vertices) for i in indices):
        raise ContractError("calibration needs explicit unique landmark indices")
    if type(regularization) not in (int,float) or not math.isfinite(regularization) or regularization <= 0:
        raise ContractError("calibration regularization must be finite and positive")
    if type(iterations) is not int or not 1 <= iterations <= 1000:
        raise ContractError("calibration iteration bound exceeded")
    numeric_target = points([list(row) for row in target],"target landmarks",len(indices))
    desired = [numeric_target[j][k]-asset.vertices[i][k] for j,i in enumerate(indices) for k in range(3)]
    columns = [[delta[i][k] for i in indices for k in range(3)] for delta in asset.deltas]
    norms = [sum(x*x for x in column)+regularization for column in columns]
    weights, residual = [0.0]*len(columns), desired.copy()
    for _ in range(iterations):
        change = 0.0
        for j,(column,norm,(low,high)) in enumerate(zip(columns,norms,asset.bounds,strict=True)):
            proposal = weights[j]+(sum(x*y for x,y in zip(column,residual,strict=True))-regularization*weights[j])/norm
            value = max(low,min(high,proposal))
            step = value-weights[j]
            residual = [y-step*x for x,y in zip(column,residual,strict=True)]
            weights[j] = value
            change = max(change,abs(step))
        if change < 1e-10:
            break
    return {"coefficients":tuple(weights),"landmark_rmse_m":math.sqrt(sum(x*x for x in residual)/len(residual)),
            "method":"bounded_coordinate_descent", "jaw_fitted":False}


def idle_coefficients(asset: MeshAsset, audio_sample: int, sample_rate: int, *, mode: str,
                      blink_channel: str | None = None, protected: tuple[str,...] = ()) -> tuple:
    """Neutral by default; optional calibrated upper-face blink, no mouth pumping."""
    from talkinghead_v2.contracts import integer
    integer(audio_sample,"audio sample"); integer(sample_rate,"sample rate",1)
    if mode not in {"idle","listening","speaking","cancelled"}:
        raise ContractError("unknown behavior state")
    result = [0.0]*len(asset.rig.coefficient_names)
    if mode != "cancelled" and blink_channel is not None:
        if blink_channel in protected or blink_channel not in asset.behavior_channels:
            raise ContractError("behavior cannot own protected or unknown articulation channels")
        # One 200 ms blink every four seconds, indexed in the audio clock.
        phase = audio_sample % (sample_rate*4)
        duration = max(1,sample_rate//5)
        if phase < duration:
            strength = 1-abs(2*phase/duration-1)
            i = asset.rig.coefficient_names.index(blink_channel)
            result[i] = max(asset.bounds[i][0],min(asset.bounds[i][1],strength))
    return tuple(result)


def compose_behavior(asset: MeshAsset, frame: MotionFrame, *, mode: str, blink_channel=None) -> MotionFrame:
    """Apply only authorized upper-channel residuals; preserve speech and rigid pose."""
    if frame.rig != asset.rig:
        raise ContractError("behavior rig identity mismatch")
    residual=idle_coefficients(asset,frame.audio_sample,frame.sample_rate,mode=mode,blink_channel=blink_channel)
    values=tuple(max(low,min(high,base+extra)) if name in asset.behavior_channels else base
        for name,base,extra,(low,high) in zip(asset.rig.coefficient_names,frame.coefficients,residual,asset.bounds,strict=True))
    return replace(frame,coefficients=values)
