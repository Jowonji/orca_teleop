"""Tests for the live WebXR and legacy recorded Quest adapters."""

import numpy as np
import pytest

from orca_teleop.ingress.metaquest.landmarks import (
    WEBXR_TO_RETARGETER_LANDMARK_INDICES,
    WristAngleEstimator,
    retargeter_landmarks_from_quest,
    retargeter_landmarks_from_webxr,
    xr_flat_matrix_to_numpy,
    xr_matrix_to_flu_matrix,
)


def test_webxr_landmarks_drop_only_extra_metacarpals():
    points = np.arange(25 * 3, dtype=np.float64).reshape(25, 3)

    converted = retargeter_landmarks_from_webxr(points, "right")

    assert converted.shape == (21, 3)
    np.testing.assert_array_equal(
        converted,
        points[list(WEBXR_TO_RETARGETER_LANDMARK_INDICES)],
    )


def test_webxr_adapter_does_not_mirror_right_hand():
    points = np.arange(25 * 3, dtype=np.float64).reshape(25, 3)

    right = retargeter_landmarks_from_webxr(points, "right")
    left = retargeter_landmarks_from_webxr(points, "left")

    np.testing.assert_array_equal(right, left)


def test_webxr_adapter_rejects_wrong_shape_and_handedness():
    with pytest.raises(ValueError, match=r"\(25, 3\)"):
        retargeter_landmarks_from_webxr(np.zeros((21, 3)), "right")
    with pytest.raises(ValueError, match="handedness"):
        retargeter_landmarks_from_webxr(np.zeros((25, 3)), "center")


def test_webxr_matrix_decoder_handles_column_major_input():
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    raw_column_major = matrix.T.ravel()

    np.testing.assert_array_equal(xr_flat_matrix_to_numpy(raw_column_major), matrix)


def test_identity_webxr_matrix_stays_identity_in_flu_basis():
    converted = xr_matrix_to_flu_matrix(np.eye(4).T.ravel())

    np.testing.assert_allclose(converted, np.eye(4))


def _wrist_matrix(pitch_degrees: float) -> np.ndarray:
    pitch = np.radians(pitch_degrees)
    matrix = np.eye(4)
    matrix[:3, 0] = [np.cos(pitch), 0.0, np.sin(pitch)]
    return matrix


def test_wrist_angle_estimator_is_relative_and_resettable():
    estimator = WristAngleEstimator()

    assert estimator.update(_wrist_matrix(10.0)) == pytest.approx(0.0)
    assert estimator.update(_wrist_matrix(-5.0)) == pytest.approx(15.0)

    estimator.reset()
    assert estimator.update(_wrist_matrix(-5.0)) == pytest.approx(0.0)


def test_right_hand_landmarks_flip_device_y_axis():
    """Legacy recorded HTS/Unity data still uses the historical mirror."""
    points = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]])

    converted = retargeter_landmarks_from_quest(points, "right")

    assert converted.tolist() == [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]


def test_left_hand_landmarks_are_unchanged():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]])

    converted = retargeter_landmarks_from_quest(points, "left")

    assert converted.tolist() == points.tolist()


def test_landmark_conversion_does_not_mutate_input():
    points = np.array([[1.0, 2.0, 3.0]])

    converted = retargeter_landmarks_from_quest(points, "right")

    assert points.tolist() == [[1.0, 2.0, 3.0]]
    assert not np.allclose(converted, points)


def test_landmark_conversion_rejects_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        retargeter_landmarks_from_quest(np.zeros((21,)), "right")


def test_landmark_conversion_rejects_invalid_handedness():
    with pytest.raises(ValueError, match="handedness"):
        retargeter_landmarks_from_quest(np.zeros((21, 3)), "center")
