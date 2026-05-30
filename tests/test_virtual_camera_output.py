import numpy as np

from virtual_camera.camera_output import VirtualCameraOutput


def test_virtual_camera_converts_chw_float_to_hwc_uint8():
    camera = VirtualCameraOutput(width=4, height=3)
    frame_chw = np.linspace(0.0, 1.0, 3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)

    converted = camera._to_uint8_rgb(frame_chw)
    assert converted.shape == (3, 4, 3)
    assert converted.dtype == np.uint8

