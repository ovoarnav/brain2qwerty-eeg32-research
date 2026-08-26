# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from brain2qwerty_v1.config.model_config import ENCODER, TRANSFORMER
from brain2qwerty_v1.config.xp_config import eeg_experiment_config
from brain2qwerty_v1.eeg import EEG32_CHANNELS, EEG61_CHANNELS
from brain2qwerty_v1.main import _windows_safe_timeline_cache_key


def test_eeg32_montage_is_fixed_and_unique():
    assert len(EEG32_CHANNELS) == 32
    assert len(set(EEG32_CHANNELS)) == 32
    assert len(EEG61_CHANNELS) == 61
    assert set(EEG32_CHANNELS) < set(EEG61_CHANNELS)
    assert EEG32_CHANNELS == (
        "F3",
        "F4",
        "F5",
        "F6",
        "Fz",
        "FC1",
        "FC2",
        "FC3",
        "FC4",
        "FC5",
        "FC6",
        "FT7",
        "FT8",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "Cz",
        "T7",
        "T8",
        "CP1",
        "CP2",
        "CP3",
        "CP4",
        "CP5",
        "CP6",
        "CPz",
        "TP7",
        "TP8",
        "Pz",
    )


def test_full_and_eeg32_change_only_the_input_montage():
    full = eeg_experiment_config("full")
    eeg32 = eeg_experiment_config("eeg32")

    assert full["data"]["study"]["name"] == "Pinet2024Eeg"
    assert full["data"]["neuro"]["name"] == "EegExtractor"
    assert full["data"]["neuro"]["picks"] == ("eeg",)
    assert eeg32["data"]["neuro"]["picks"] == EEG32_CHANNELS
    assert full["brain_model_config"] == eeg32["brain_model_config"] == ENCODER
    assert full["transformer_config"] == eeg32["transformer_config"] == TRANSFORMER

    full["data"]["neuro"]["picks"] = EEG32_CHANNELS
    full["output_dir"] = eeg32["output_dir"]
    assert full == eeg32


def test_windows_timeline_cache_key_is_stable_and_filename_safe():
    timeline = {"session": 1, "subject": "002", "task": "task1"}
    key = _windows_safe_timeline_cache_key(timeline)
    assert key == _windows_safe_timeline_cache_key(dict(reversed(timeline.items())))
    assert not set('<>:"/\\|?*').intersection(key)
