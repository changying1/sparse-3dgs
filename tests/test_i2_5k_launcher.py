from pathlib import Path

from tools.run_i2_d1_k20_5k import ARMS, SOURCE, command_for


def _normalized_common(command):
    command = list(command)
    if "--i2_active_plan" in command:
        command = command[:command.index("--i2_active_plan")]
    command[command.index("-m") + 1] = "<OUTPUT>"
    return command


def _values_after(command, option, count):
    start = command.index(option) + 1
    return command[start:start + count]


def test_three_5k_arms_share_all_non_intervention_training_arguments(tmp_path):
    commands = {
        arm: command_for(arm, tmp_path / directory)
        for arm, directory in ARMS.items()
    }
    expected = _normalized_common(commands["control"])
    assert _normalized_common(commands["release"]) == expected
    assert _normalized_common(commands["reallocation"]) == expected


def test_5k_commands_start_from_frozen_1201_and_use_requested_schedule(tmp_path):
    for arm, directory in ARMS.items():
        command = command_for(arm, tmp_path / directory)
        assert _values_after(command, "--iterations", 1) == ["5000"]
        assert _values_after(command, "--start_checkpoint", 1) == [str(SOURCE)]
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
        assert "chkpnt1252.pth" not in " ".join(command)


def test_5k_intervention_flags_are_arm_specific(tmp_path):
    control = command_for("control", tmp_path / "control")
    release = command_for("release", tmp_path / "release")
    reallocation = command_for("reallocation", tmp_path / "reallocation")
    assert "--enable_i2_active_pilot" not in control
    assert "--enable_i2_release_only" not in control
    assert "--enable_i2_release_only" in release
    assert "--enable_i2_active_pilot" not in release
    assert "--enable_i2_active_pilot" in reallocation
    assert "--enable_i2_release_only" not in reallocation
