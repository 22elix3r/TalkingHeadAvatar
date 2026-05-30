# Virtual Camera Module

This module injects rendered avatar frames into a v4l2loopback virtual camera device for use in video conferencing applications.

## Components
- `camera_output.py` — pyvirtualcam-based frame writer (512×512 @ 30 FPS)

## Data Flow
```
Rendered frame (3, H, W) float → uint8 conversion → pyvirtualcam → /dev/video10 → Zoom/Meet/Teams
```

## Prerequisites
- v4l2loopback kernel module loaded
- `pyvirtualcam` pip package installed

## OBS Black-Screen Troubleshooting
1. Confirm the producer is actually running: `python run_demo.py ... --video_only` should print `[VirtualCameraOutput] Opened ...`.
2. If OBS shows black on `/dev/video10`, reload loopback with non-exclusive caps for the avatar device (OBS reader side is more reliable with this):
   - `sudo OBS_EXCLUSIVE_CAPS=1 AVATAR_EXCLUSIVE_CAPS=0 ./scripts/setup_v4l2loopback.sh`
3. Ensure OBS source points to `/dev/video10` (avatar feed), not `/dev/video0` (OBS virtual camera output sink).

## See Also
- Phase 9 skill: `.agent/skills/phase9_virtual_camera/skill.md`
- Implementation plan: `implementation_plan.md` (Phase 9)
