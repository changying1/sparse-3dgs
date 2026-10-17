import json
from pathlib import Path

from tools.run_i2_receiver_v2_5k import command as receiver_v2_5k_command
from tools.run_i2_v2_selection_anchor_clone_1252 import command as anchor_short_command
from tools.run_i2_v2_selection_anchor_clone_5k import command as anchor_5k_command


def _normalized(command):
    result = list(command)
    result[result.index("-m") + 1] = "<OUTPUT>"
    result[result.index("--i2_active_plan") + 1] = "<PLAN>"
    result[result.index("--i2_active_event_log") + 1] = "<EVENT>"
    return result


def _values_after(command, option, count):
    start = command.index(option) + 1
    return command[start:start + count]


def test_anchor_clone_short_differs_from_receiver_v2_only_by_output_and_plan():
    root = Path(__file__).resolve().parents[1]
    receiver_v2 = json.loads(
        (root / "output/I2_D1_ReceiverV2_K20_1252/launch_meta.json").read_text(
            encoding="utf-8"
        )
    )["argv"]
    assert _normalized(anchor_short_command()) == _normalized(receiver_v2)


def test_anchor_clone_5k_differs_from_receiver_v2_only_by_output_and_plan():
    assert _normalized(anchor_5k_command()) == _normalized(receiver_v2_5k_command())


def test_anchor_clone_5k_schedule_and_tracking_contract():
    command = anchor_5k_command()
    assert _values_after(command, "--iterations", 1) == ["5000"]
    assert _values_after(command, "--test_iterations", 5) == [
        "1500", "2000", "3000", "4000", "5000",
    ]
    assert _values_after(command, "--checkpoint_iterations", 3) == [
        "2000", "3500", "5000",
    ]
    assert _values_after(command, "--save_iterations", 1) == ["5000"]
    assert "--enable_i2_lifecycle_tracking" in command
    assert "--enable_i2_observation_tracking" in command
    assert "--enable_i2_future_utility_tracking" not in command
    assert "--enable_oe_structural_loss" not in command
