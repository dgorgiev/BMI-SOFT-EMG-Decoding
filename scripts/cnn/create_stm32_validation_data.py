"""Create deterministic held-out validation tensors for the STM32 split model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from bmiemg.data.epn612.loader import load_user_trials
from bmiemg.preprocessing.emg import ChannelNormalizationStats, normalize_emg_channels
from scripts.cnn.export_executorch import (
    DEFAULT_CHECKPOINT,
    WindowEncoder,
    load_board_checkpoint,
)
from scripts.epn612.adaptation import BoardAdaptationConfig
from scripts.epn612.dataset import (
    EPN612DatasetConfig,
    WindowConfig,
    extract_trial_windows,
)
from scripts.epn612.split import select_participant_splits

DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "artifacts" / "cnn" / "stm32_validation"


def file_sha256(path: Path) -> str:
    """Return a stable identifier for a source artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def create_validation_data(
    checkpoint_path: Path,
    output_directory: Path,
    samples_per_class: int,
    split: str = "test",
) -> None:
    """Create balanced sequences from the selected participant split."""

    checkpoint, model = load_board_checkpoint(checkpoint_path)
    experiment = checkpoint["experiment_config"]
    data_values = experiment["data"]
    window_values = data_values["windows"]
    sequence_length = int(window_values["sequence_length"])
    class_labels = tuple(checkpoint["class_labels"])
    label_to_index = {label: index for index, label in enumerate(class_labels)}

    adaptation = BoardAdaptationConfig(**data_values["adaptation"])
    windows = WindowConfig(**window_values)
    dataset_root = Path(data_values["root"])
    dataset_config = EPN612DatasetConfig(
        root=dataset_root,
        participant_id=data_values["participant_id"],
        max_participants=data_values["max_participants"],
        split_ratios=tuple(data_values["split_ratios"]),
        class_labels=tuple(data_values["class_labels"]),
        normalization_epsilon=float(data_values["normalization_epsilon"]),
        adaptation=adaptation,
        windows=windows,
    )

    normalization_values = checkpoint["input"]["normalization"]
    normalization = ChannelNormalizationStats(
        mean=tuple(normalization_values["mean"]),
        standard_deviation=tuple(normalization_values["standard_deviation"]),
        epsilon=float(normalization_values["epsilon"]),
    )

    train_ratio, validation_ratio, test_ratio = dataset_config.split_ratios
    selection = select_participant_splits(
        dataset_root,
        max_participants=dataset_config.max_participants,
        ratios={
            "train": train_ratio,
            "val": validation_ratio,
            "test": test_ratio,
        },
        random_seed=int(experiment["random_seed"]),
    )
    split_participants = [
        participant
        for participant in selection.participants
        if selection.assignments[participant.participant_id] == split
    ]

    selected_sequences: list[np.ndarray] = []
    selected_labels: list[int] = []
    manifest_rows: list[dict[str, str | int]] = []
    counts = {label: 0 for label in class_labels}

    for participant in split_participants:
        participant_counts = dict.fromkeys(class_labels, 0)
        for trial in load_user_trials(
            participant.source_path, section="trainingSamples"
        ):
            if trial.gesture not in label_to_index:
                continue
            label = trial.gesture
            if counts[label] >= samples_per_class:
                continue

            trial_windows = extract_trial_windows(trial, dataset_config)
            if len(trial_windows) < sequence_length:
                continue
            normalized_windows = normalize_emg_channels(
                np.stack(trial_windows),
                normalization,
            )
            for end_index in range(sequence_length - 1, len(normalized_windows)):
                if counts[label] >= samples_per_class:
                    break
                if split == "train" and participant_counts[label] >= max(
                    1, samples_per_class // 5
                ):
                    break
                sequence = normalized_windows[
                    end_index - sequence_length + 1 : end_index + 1
                ]
                selected_sequences.append(np.asarray(sequence, dtype=np.float32))
                selected_labels.append(label_to_index[label])
                manifest_rows.append(
                    {
                        "sample_index": len(selected_sequences) - 1,
                        "label_index": label_to_index[label],
                        "label": label,
                        "participant_id": participant.participant_id,
                        "trial_id": trial.trial_id,
                        "sequence_end_window": end_index,
                    }
                )
                counts[label] += 1
                participant_counts[label] += 1

        if all(count == samples_per_class for count in counts.values()):
            break

    incomplete = {
        label: count for label, count in counts.items() if count != samples_per_class
    }
    if incomplete:
        raise RuntimeError(f"Could not create a balanced validation set: {incomplete}")

    sequence_inputs = np.stack(selected_sequences).astype(np.float32, copy=False)
    labels = np.asarray(selected_labels, dtype=np.int64)
    encoder_inputs = sequence_inputs.reshape(
        -1,
        sequence_inputs.shape[-2],
        sequence_inputs.shape[-1],
    )

    encoder_outputs = (
        WindowEncoder(model).eval()(torch.from_numpy(encoder_inputs)).numpy()
    )
    classifier_inputs = encoder_outputs.reshape(
        len(sequence_inputs), sequence_length, -1
    )
    assert model.sequence_gru is not None and model.sequence_classifier is not None
    _, hidden = model.sequence_gru(torch.from_numpy(classifier_inputs))
    classifier_outputs = model.sequence_classifier(hidden[-1]).numpy()
    predictions = np.argmax(classifier_outputs, axis=1).astype(np.int64)

    output_directory.mkdir(parents=True, exist_ok=True)
    np.save(output_directory / "cnn_encoder_inputs.npy", encoder_inputs)
    np.save(output_directory / "cnn_encoder_reference_outputs.npy", encoder_outputs)
    np.save(output_directory / "gru_classifier_inputs.npy", classifier_inputs)
    np.save(
        output_directory / "gru_classifier_reference_outputs.npy", classifier_outputs
    )
    np.save(output_directory / "sequence_inputs.npy", sequence_inputs)
    np.save(output_directory / "sequence_reference_outputs.npy", classifier_outputs)
    np.save(output_directory / "labels.npy", labels)
    np.save(output_directory / "predictions.npy", predictions)

    with (output_directory / "manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "description": "Balanced subject-disjoint inputs and PyTorch reference outputs for the STM32 decoder.",
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "class_labels": class_labels,
        "samples_per_class": samples_per_class,
        "sequence_count": len(sequence_inputs),
        "sequence_shape": list(sequence_inputs.shape),
        "encoder_input_shape": list(encoder_inputs.shape),
        "encoder_output_shape": list(encoder_outputs.shape),
        "classifier_input_shape": list(classifier_inputs.shape),
        "classifier_output_shape": list(classifier_outputs.shape),
        "dtype": "float32",
        "sampling_rate_hz": checkpoint["input"]["sampling_rate_hz"],
        "window_size_samples": window_values["size_samples"],
        "window_step_samples": window_values["step_samples"],
        "sequence_length": sequence_length,
        "normalization": normalization_values,
        "label_accuracy_of_reference_model": float(np.mean(predictions == labels)),
    }
    with (output_directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(f"Created validation data in {output_directory}")
    print(f"Sequence inputs: {sequence_inputs.shape} {sequence_inputs.dtype}")
    print(f"Encoder inputs/outputs: {encoder_inputs.shape} -> {encoder_outputs.shape}")
    print(
        "Classifier inputs/outputs: "
        f"{classifier_inputs.shape} -> {classifier_outputs.shape}"
    )
    print(f"Class counts: {counts}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    args = parser.parse_args()
    if args.samples_per_class <= 0:
        parser.error("--samples-per-class must be positive")
    return args


def main() -> None:
    """Create the validation artifacts."""

    args = parse_args()
    create_validation_data(
        args.checkpoint.resolve(),
        args.output_directory.resolve(),
        args.samples_per_class,
        args.split,
    )


if __name__ == "__main__":
    main()
