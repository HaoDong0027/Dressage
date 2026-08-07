"""Transport integration for Dressage."""

__all__ = [
    "TITOState",
    "TrainingPayloadRef",
    "TrajectoryBuildConfig",
    "TrajectoryBuilder",
    "TrajectoryManifest",
    "TransferQueueRuntime",
    "TransferQueueTrajectoryStore",
    "clear_training_batch",
    "prepare_trajectory",
    "register_training_batch",
    "start_transport_coordinator",
]


def __getattr__(name: str):
    if name in {"TrajectoryBuildConfig", "TrajectoryBuilder"}:
        from .builder import TrajectoryBuildConfig, TrajectoryBuilder

        return {
            "TrajectoryBuildConfig": TrajectoryBuildConfig,
            "TrajectoryBuilder": TrajectoryBuilder,
        }[name]
    if name in {
        "clear_training_batch",
        "prepare_trajectory",
        "register_training_batch",
    }:
        from .client import (
            clear_training_batch,
            prepare_trajectory,
            register_training_batch,
        )

        return {
            "clear_training_batch": clear_training_batch,
            "prepare_trajectory": prepare_trajectory,
            "register_training_batch": register_training_batch,
        }[name]
    if name == "TrajectoryManifest":
        from .manifest import TrajectoryManifest

        return TrajectoryManifest
    if name == "TrainingPayloadRef":
        from .payload import TrainingPayloadRef

        return TrainingPayloadRef
    if name in {"TITOState", "TransferQueueRuntime"}:
        from .runtime import TITOState, TransferQueueRuntime

        return {
            "TITOState": TITOState,
            "TransferQueueRuntime": TransferQueueRuntime,
        }[name]
    if name == "TransferQueueTrajectoryStore":
        from .store import TransferQueueTrajectoryStore

        return TransferQueueTrajectoryStore
    if name == "start_transport_coordinator":
        from .assembler import start_transport_coordinator

        return start_transport_coordinator
    raise AttributeError(name)
