from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_kb_fields(path: Path, names: set[str]) -> dict[str, int]:
    try:
        lines = path.read_text().splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return {}
    values: dict[str, int] = {}
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if separator and key in names:
            values[key] = int(raw_value.split()[0])
    return values


def _read_process_memory(pid: int) -> dict[str, Any] | None:
    status = _read_kb_fields(
        Path(f"/proc/{pid}/status"),
        {"VmRSS", "VmHWM"},
    )
    if not status:
        return None
    smaps = _read_kb_fields(
        Path(f"/proc/{pid}/smaps_rollup"),
        {"Pss"},
    )
    return {
        "pid": pid,
        "hostname": socket.gethostname(),
        "rss_bytes": status.get("VmRSS", 0) * 1024,
        "pss_bytes": smaps.get("Pss", 0) * 1024,
        "peak_rss_bytes": status.get("VmHWM", 0) * 1024,
    }


def _read_proxy_memory(pid_file: Path) -> dict[str, Any] | None:
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return _read_process_memory(pid)


def _actor_category(actor: Any) -> str | None:
    name = actor.name or ""
    if actor.class_name == "RolloutManager":
        return "rollout_manager"
    if name == "TransferQueueController":
        return "controller"
    if name.startswith("TransferQueueStorageUnit#"):
        return "storage_units"
    if name == "DressageTransportCoordinator":
        return "coordinator"
    if name.startswith("DressageTrajectoryAssembler#"):
        return "assemblers"
    return None


def _memory_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "actor_count": len(items),
        "rss_bytes": sum(item["rss_bytes"] for item in items),
        "pss_bytes": sum(item["pss_bytes"] for item in items),
        "peak_rss_bytes_sum": sum(item["peak_rss_bytes"] for item in items),
        "actors": items,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-pid-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from ray.util.state import list_actors

    ray.init(address="auto", log_to_driver=False)
    read_process_memory = ray.remote(num_cpus=0)(_read_process_memory)
    stop_event = threading.Event()

    def stop(*_: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started_at = time.monotonic()
    proxy_pid_file = Path(args.proxy_pid_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_file.open("a", encoding="utf-8", buffering=1) as output:
            while not stop_event.is_set():
                sample_started_at = time.monotonic()
                actors = [
                    actor
                    for actor in list_actors(
                        filters=[("state", "=", "ALIVE")],
                        limit=10000,
                        detail=True,
                        raise_on_missing_output=False,
                    )
                    if _actor_category(actor) is not None
                    and actor.pid
                    and actor.node_id
                ]
                actor_refs = [
                    read_process_memory.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=actor.node_id,
                            soft=False,
                        )
                    ).remote(int(actor.pid))
                    for actor in actors
                ]
                actor_memory = ray.get(actor_refs) if actor_refs else []
                actor_processes = [
                    {
                        "name": actor.name or actor.class_name,
                        "node_id": actor.node_id,
                        "category": _actor_category(actor),
                        **memory,
                    }
                    for actor, memory in zip(actors, actor_memory, strict=True)
                    if memory is not None
                ]
                grouped = {
                    category: [
                        item for item in actor_processes if item["category"] == category
                    ]
                    for category in (
                        "rollout_manager",
                        "controller",
                        "storage_units",
                        "coordinator",
                        "assemblers",
                    )
                }
                cluster_resources = ray.cluster_resources()
                available_resources = ray.available_resources()
                object_store_capacity = int(
                    cluster_resources.get("object_store_memory", 0)
                )
                object_store_available = int(
                    available_resources.get("object_store_memory", 0)
                )
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": round(time.monotonic() - started_at, 3),
                    "monitor_pid": os.getpid(),
                    "proxy": _read_proxy_memory(proxy_pid_file),
                    "rollout_manager": _memory_group(grouped["rollout_manager"]),
                    "transfer_queue": {
                        "controller": _memory_group(grouped["controller"]),
                        "storage_units": _memory_group(grouped["storage_units"]),
                    },
                    "transport": {
                        "coordinator": _memory_group(grouped["coordinator"]),
                        "assemblers": _memory_group(grouped["assemblers"]),
                    },
                    "ray_object_store": {
                        "capacity_bytes": object_store_capacity,
                        "available_bytes": object_store_available,
                        "used_bytes": max(
                            0,
                            object_store_capacity - object_store_available,
                        ),
                    },
                }
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                remaining = args.interval_seconds - (
                    time.monotonic() - sample_started_at
                )
                if remaining > 0:
                    stop_event.wait(remaining)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
