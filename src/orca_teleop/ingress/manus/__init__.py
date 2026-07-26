"""Manus MetaGloves Pro ingress support."""

__all__ = [
    "MANUS_MANO_REPO_ID",
    "ManusDatasetPublisher",
    "extract_mano_keypoints_from_manus_csv",
    "manus_unity_positions_to_mano_keypoints",
]


def __getattr__(name: str):
    if name in __all__:
        from orca_teleop.ingress.manus import dummy_publisher

        return getattr(dummy_publisher, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
