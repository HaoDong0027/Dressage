from types import SimpleNamespace

from dressage.transport.memory_monitor import _actor_category, _read_node_memory


def test_rollout_manager_actor_category_uses_class_name():
    actor = SimpleNamespace(name="", class_name="RolloutManager")

    assert _actor_category(actor) == "rollout_manager"


def test_read_node_memory(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       1000 kB\n"
        "MemAvailable:    250 kB\n"
        "SwapTotal:       100 kB\n"
        "SwapFree:         40 kB\n"
    )

    memory = _read_node_memory(meminfo)

    assert memory is not None
    assert memory["total_bytes"] == 1000 * 1024
    assert memory["available_bytes"] == 250 * 1024
    assert memory["used_bytes"] == 750 * 1024
    assert memory["usage_percent"] == 75.0
    assert memory["swap_total_bytes"] == 100 * 1024
    assert memory["swap_used_bytes"] == 60 * 1024
