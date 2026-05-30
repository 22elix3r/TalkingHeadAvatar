import numpy as np

from virtual_camera.camera_output import VirtualCameraOutput


def test_virtual_camera_converts_chw_float_to_hwc_uint8():
    camera = VirtualCameraOutput(width=4, height=3)
    frame_chw = np.linspace(0.0, 1.0, 3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)

    converted = camera._to_uint8_rgb(frame_chw)
    assert converted.shape == (3, 4, 3)
    assert converted.dtype == np.uint8


def test_virtual_camera_letterboxes_mismatched_frame_size():
    camera = VirtualCameraOutput(width=8, height=4)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 1] = 255

    converted = camera._to_uint8_rgb(frame)
    assert converted.shape == (4, 8, 3)
    assert converted.dtype == np.uint8
    assert np.all(converted[:, 2:6, 1] == 255)
    assert np.all(converted[:, :2] == 255)
    assert np.all(converted[:, 6:] == 255)


def test_virtual_camera_converts_rgb_to_i420_flat_buffer():
    camera = VirtualCameraOutput(width=4, height=2)
    frame = np.full((2, 4, 3), 255, dtype=np.uint8)

    converted = camera._rgb_to_i420(frame)
    assert converted.shape == (4 * 2 * 3 // 2,)
    assert converted.dtype == np.uint8
    assert np.all(converted[:8] >= 230)
