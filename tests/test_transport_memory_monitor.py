from types import SimpleNamespace

from dressage.transport.memory_monitor import _actor_category


def test_rollout_manager_actor_category_uses_class_name():
    actor = SimpleNamespace(name="", class_name="RolloutManager")

    assert _actor_category(actor) == "rollout_manager"
