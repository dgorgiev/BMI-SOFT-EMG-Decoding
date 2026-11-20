"""Export and verify the two batch-one ExecuTorch models for the WBA65."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.cnn.cnn import TCNN
from scripts.cnn.training import TCNNConfig, build_tcnn

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts/cnn"
DEFAULT_CHECKPOINT = ARTIFACTS / (
    "CNNv1_50users_200mswindow_raw_no-notch_rotation_"
    "regularized-wd0p0001-dropout0p3_sequence5-gru64.pt"
)
EXECUTORCH_VERSION = "1.6.0"
BOARD_INPUT_SHAPE = (1, 5, 8, 400)
BOARD_CLASS_LABELS = ("noGesture", "fist", "waveIn", "waveOut", "open", "pinch")
EMBEDDING_SIZE = 64


class WindowEncoder(nn.Module):
    """Return one embedding per EMG window."""

    def __init__(self, model: TCNN) -> None:
        super().__init__()
        self.model = model

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return self.model._window_feature_maps(window).mean(dim=(2, 3))


class SequenceClassifier(nn.Module):
    """Lower the checkpoint GRU to portable linear and elementwise operators."""

    def __init__(self, gru: nn.GRU, classifier: nn.Linear) -> None:
        super().__init__()
        self.gru = gru
        self.classifier = classifier

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # PyTorch stores the GRU gates in reset, update, new order.
        projected = F.linear(embeddings, self.gru.weight_ih_l0, self.gru.bias_ih_l0)
        hidden = embeddings.new_zeros((embeddings.shape[0], self.gru.hidden_size))
        for step in range(embeddings.shape[1]):
            input_reset, input_update, input_new = projected[:, step].chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = F.linear(
                hidden, self.gru.weight_hh_l0, self.gru.bias_hh_l0
            ).chunk(3, dim=-1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            candidate = torch.tanh(input_new + reset * hidden_new)
            hidden = (1.0 - update) * candidate + update * hidden
        return self.classifier(hidden)


def load_board_checkpoint(path: Path) -> tuple[dict[str, Any], TCNN]:
    """Load a trusted checkpoint and reject incompatible preprocessing or I/O."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    input_config = checkpoint["input"]
    shape = tuple(1 if size == "batch" else size for size in input_config["shape"])
    data = checkpoint["experiment_config"]["data"]
    adaptation = data["adaptation"]
    labels = tuple(checkpoint["class_labels"])
    if (
        shape != BOARD_INPUT_SHAPE
        or input_config["dtype"] != "float32"
        or input_config["sampling_rate_hz"] != 2000
        or data["windows"]["step_samples"] != 200
        or adaptation["target_adc_bits"] != 12
        or adaptation["rectify"]
        or adaptation["notch"]
        or labels != BOARD_CLASS_LABELS
    ):
        raise ValueError(
            "Checkpoint input, preprocessing or class order differs from the WBA65 decoder"
        )

    normalization = input_config["normalization"]
    means = normalization["mean"]
    deviations = normalization["standard_deviation"]
    epsilon = normalization["epsilon"]
    if (
        normalization["type"] != "per_channel_standardization"
        or len(means) != shape[2]
        or len(deviations) != shape[2]
        or not all(math.isfinite(x) for x in (*means, *deviations, epsilon))
        or any(x < 0 for x in deviations)
        or epsilon <= 0
    ):
        raise ValueError("Checkpoint has invalid channel normalization")

    model = build_tcnn(TCNNConfig(**checkpoint["model_config"]), n_classes=len(labels))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if (
        model.sequence_gru is None
        or model.sequence_classifier is None
        or model.sequence_gru.input_size != EMBEDDING_SIZE
    ):
        raise ValueError(
            "WBA65 deployment requires the sequence GRU with 64 CNN features"
        )
    return checkpoint, model.eval()


def _float_array(values) -> str:
    return ", ".join(f"{float(value):.9e}f" for value in values)


class CortexMReference(torch.fx.Interpreter):
    """Run the lowered host graph, including dilated integer convolution."""

    def call_function(self, target, args, kwargs):
        from executorch.exir.dialects._ops import ops

        if target == ops.edge.cortex_m.quantized_conv2d_nhwc.default:
            return self.quantized_conv2d(*args)
        if target == ops.edge.cortex_m.quantized_avg_pool2d_nhwc.default:
            inputs, kernel, _, padding, _, _, _, _, _ = args
            if tuple(kernel) != tuple(inputs.shape[1:3]) or any(padding):
                raise ValueError("Expected an unpadded global average pool")
            # CMSIS-NN rounds signed integer pool sums to nearest, ties away
            # from zero. PyTorch's default emulation uses float round-to-even.
            count = kernel[0] * kernel[1]
            sums = inputs.sum(dim=(1, 2), keepdim=True, dtype=torch.int32)
            adjusted = sums + torch.where(sums > 0, count // 2, -(count // 2))
            return torch.div(adjusted, count, rounding_mode="trunc").to(torch.int8)
        if target == ops.edge.cortex_m.quantize_per_tensor.default:
            inputs, scale, zero_point, minimum, maximum, dtype = args
            inverse_scale = torch.tensor(scale, dtype=torch.float32).reciprocal()
            return (
                ((inputs * inverse_scale).round() + zero_point)
                .clamp(minimum, maximum)
                .to(dtype)
            )
        return super().call_function(target, args, kwargs)

    @staticmethod
    def quantized_conv2d(
        inputs,
        weight,
        bias,
        stride,
        padding,
        dilation,
        input_offset,
        output_offset,
        multipliers,
        shifts,
        activation_min,
        activation_max,
        _scratch,
    ):
        from executorch.backends.cortex_m.utils import requantize_cmsis

        # PyTorch's CPU convolution does not support dilated int32. FP64
        # represents every int32 accumulation here exactly.
        nchw = inputs.permute(0, 3, 1, 2).to(torch.float64) + input_offset
        top, left = padding[:2]
        bottom, right = padding[2:] if len(padding) == 4 else padding
        padded = F.pad(nchw, (left, right, top, bottom))
        accumulators = F.conv2d(
            padded,
            weight.permute(0, 3, 1, 2).to(torch.float64),
            None if bias is None else bias.to(torch.float64),
            stride=stride,
            dilation=dilation,
            groups=nchw.shape[1] // weight.shape[3],
        ).to(torch.int32)
        requantized = torch.stack(
            [
                requantize_cmsis(accumulators[:, channel], int(multiplier), int(shift))
                for channel, (multiplier, shift) in enumerate(
                    zip(multipliers, shifts, strict=True)
                )
            ],
            dim=-1,
        )
        return (
            (requantized + output_offset)
            .clamp(activation_min, activation_max)
            .to(torch.int8)
        )


def quantize_encoder(encoder, sample, calibration_dir: Path, checkpoint_hash: str):
    """Calibrate the encoder on training subjects and lower it for Cortex-M33."""
    from executorch.backends.cortex_m.edge_compile_config import (
        cortex_m_edge_compile_config,
    )
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
        CortexMPassManager,
    )
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
    from executorch.exir import to_edge
    from executorch.exir.schema import KernelCall
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    metadata = json.loads((calibration_dir / "metadata.json").read_text())
    if (
        metadata.get("split") != "train"
        or metadata.get("checkpoint_sha256") != checkpoint_hash
    ):
        raise ValueError(
            "Calibration must use training-subject windows from the same checkpoint"
        )
    windows = np.load(calibration_dir / "cnn_encoder_inputs.npy", allow_pickle=False)
    if (
        windows.ndim != 3
        or windows.shape[1:] != BOARD_INPUT_SHAPE[2:]
        or len(windows) == 0
        or windows.dtype != np.float32
        or not np.isfinite(windows).all()
    ):
        raise ValueError(
            "Calibration requires finite float32 windows of shape (N, 8, 400)"
        )
    prepared = prepare_pt2e(
        torch.export.export(encoder, (sample,)).module(),
        CortexMQuantizer(use_explicit_layout=True),
    )
    for window in torch.from_numpy(windows):
        prepared(window.unsqueeze(0))
    converted = convert_pt2e(prepared)
    edge = to_edge(
        torch.export.export(converted, (sample,)),
        compile_config=cortex_m_edge_compile_config(),
    )
    edge = edge.transform(
        CortexMPassManager(
            target_config=CortexMTargetConfig(cpu=CortexM.M33), use_explicit_layout=True
        )
    )
    reference = CortexMReference(edge.exported_program().module())
    program = edge.to_executorch()
    plan = program.executorch_program.execution_plan[0]
    kernel_calls = [
        plan.operators[instruction.instr_args.op_index].name
        for chain in plan.chains
        for instruction in chain.instructions
        if isinstance(instruction.instr_args, KernelCall)
    ]
    convolutions = [
        name
        for name in kernel_calls
        if name.startswith(
            ("cortex_m::quantized_conv2d", "cortex_m::quantized_depthwise_conv2d")
        )
    ]
    if len(convolutions) != 4 or "aten::convolution" in kernel_calls:
        raise RuntimeError(
            f"Expected four CMSIS-NN INT8 convolutions, got {kernel_calls}"
        )
    print(
        f"Calibrated on {len(windows)} training windows. Cortex-M conv kernels: {convolutions}"
    )
    return program, reference


@torch.no_grad()
def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    validation_inputs: Path | None,
    calibration_dir: Path | None = None,
    int8_replay_dir: Path | None = None,
) -> None:
    """Verify the split pipeline on host before writing the firmware artifacts."""
    if version("executorch") != EXECUTORCH_VERSION:
        raise RuntimeError(
            f"Install executorch=={EXECUTORCH_VERSION} for this firmware"
        )
    from executorch.exir import to_edge
    from executorch.runtime import Runtime

    checkpoint, model = load_board_checkpoint(checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if calibration_dir is not None and (
        validation_inputs is None or int8_replay_dir is None
    ):
        raise ValueError(
            "INT8 export requires held-out inputs and an INT8 replay output directory"
        )
    encoder = WindowEncoder(model).eval()
    classifier = SequenceClassifier(
        model.sequence_gru, model.sequence_classifier
    ).eval()
    generator = torch.Generator().manual_seed(0)
    examples = torch.randn((3, *BOARD_INPUT_SHAPE[1:]), generator=generator)
    if validation_inputs is not None:
        held_out = torch.from_numpy(np.load(validation_inputs, allow_pickle=False))
        if (
            held_out.shape[1:] != BOARD_INPUT_SHAPE[1:]
            or held_out.shape[0] == 0
            or held_out.dtype != torch.float32
            or not torch.isfinite(held_out).all()
        ):
            raise ValueError(
                "Validation input must be finite float32 with shape (N, 5, 8, 400)"
            )
        examples = torch.cat((examples, held_out))

    programs = []
    buffers = {}
    methods = {}
    arena_sizes = {}
    operators = set()
    runtime = Runtime.get()
    for name, module, sample in (
        ("cnn_encoder", encoder, examples[0, :1].contiguous()),
        ("gru_classifier", classifier, torch.zeros((1, 5, EMBEDDING_SIZE))),
    ):
        if calibration_dir is not None and name == "cnn_encoder":
            program, quantized_reference = quantize_encoder(
                encoder, sample, calibration_dir, checkpoint_hash
            )
        else:
            exported = torch.export.export(module, (sample,), strict=True)
            program = to_edge(exported).to_executorch()
        (plan,) = program.executorch_program.execution_plan
        reserved, arena_size = plan.non_const_buffer_sizes
        if reserved != 0 or arena_size <= 0:
            raise RuntimeError(f"{name} has an unexpected memory plan")
        operators.update(
            f"{op.name}.{op.overload}" if op.overload else op.name
            for op in plan.operators
            if op.name.startswith("aten::")
        )
        buffers[name] = bytes(program.buffer)
        if calibration_dir is None or name != "cnn_encoder":
            loaded_program = runtime.load_program(buffers[name])
            programs.append(loaded_program)
            methods[name] = loaded_program.load_method("forward")
        arena_sizes[name] = arena_size

    if calibration_dir is not None:
        assert validation_inputs is not None and int8_replay_dir is not None
        validation_dir = validation_inputs.parent
        metadata = json.loads((validation_dir / "metadata.json").read_text())
        if (
            metadata.get("split", "test") != "test"
            or metadata["checkpoint_sha256"] != checkpoint_hash
        ):
            raise ValueError(
                "Use held-out data from the same checkpoint for validation"
            )
        labels = np.load(validation_dir / "labels.npy", allow_pickle=False)
        if labels.shape != (len(held_out),):
            raise ValueError("Held-out labels must match the number of sequences")
        references = []
        for sequence in held_out:
            embeddings = torch.cat(
                [quantized_reference.run(window.unsqueeze(0)) for window in sequence]
            )
            (logits,) = methods["gru_classifier"].execute([embeddings.unsqueeze(0)])
            references.append(logits)
        logits = torch.cat(references).numpy()
        fp32_logits = model(held_out).numpy()
        print(f"Held-out FP32 accuracy {(fp32_logits.argmax(1) == labels).mean():.1%}")
        print(f"Held-out INT8 accuracy {(logits.argmax(1) == labels).mean():.1%}")
        print(
            f"FP32/INT8 class agreement {(logits.argmax(1) == fp32_logits.argmax(1)).mean():.1%}"
        )
        int8_replay_dir.mkdir(parents=True, exist_ok=True)
        np.save(int8_replay_dir / "sequence_inputs.npy", held_out.numpy())
        np.save(int8_replay_dir / "sequence_reference_outputs.npy", logits)
        np.save(int8_replay_dir / "labels.npy", labels)
        (int8_replay_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    else:
        max_error = 0.0
        for sequence in examples:
            embeddings = torch.cat(
                [
                    methods["cnn_encoder"].execute([window.unsqueeze(0)])[0]
                    for window in sequence
                ]
            )
            torch.testing.assert_close(
                embeddings, encoder(sequence), rtol=2e-4, atol=2e-5
            )
            (logits,) = methods["gru_classifier"].execute([embeddings.unsqueeze(0)])
            expected_logits = model(sequence.unsqueeze(0))
            torch.testing.assert_close(logits, expected_logits, rtol=2e-4, atol=5e-5)
            if logits.argmax().item() != expected_logits.argmax().item():
                raise RuntimeError("ExecuTorch changed the predicted class")
            max_error = max(max_error, (logits - expected_logits).abs().max().item())

        print(
            f"Verified {len(examples)} sequences. Max absolute logit error {max_error:.3e}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, buffer in buffers.items():
        (output_dir / f"{name}.pte").write_bytes(buffer)

    normalization = checkpoint["input"]["normalization"]
    scales = [x + normalization["epsilon"] for x in normalization["standard_deviation"]]
    (output_dir / "emg_model_config.h").write_text(
        f"""// Generated from checkpoint SHA-256 {checkpoint_hash}
#pragma once

#include <cstddef>

namespace EmgModel
{{
inline constexpr std::size_t kEncoderArenaSize = {arena_sizes["cnn_encoder"]};
inline constexpr std::size_t kClassifierArenaSize = {arena_sizes["gru_classifier"]};
inline constexpr float kChannelMean[] = {{{_float_array(normalization["mean"])}}};
inline constexpr float kChannelScale[] = {{{_float_array(scales)}}};
}} // namespace EmgModel
""",
        encoding="utf-8",
    )
    (output_dir / "emg_model_ops.cmake").write_text(
        f'set(EMG_MODEL_EXECUTORCH_VERSION "{EXECUTORCH_VERSION}")\n'
        f"set(EMG_MODEL_OPS {json.dumps(';'.join(sorted(operators)))})\n"
        f"set(EMG_MODEL_CORTEX_M {'ON' if calibration_dir is not None else 'OFF'})\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir} with planned arenas {arena_sizes}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validation-inputs", type=Path, help="Optional held-out sequence_inputs.npy"
    )
    parser.add_argument(
        "--calibration-dir", type=Path, help="Training-split calibration data directory"
    )
    parser.add_argument(
        "--int8-replay-dir", type=Path, help="Destination for INT8 reference fixtures"
    )
    args = parser.parse_args()
    default_output = ARTIFACTS / (
        "executorch_int8" if args.calibration_dir else "executorch"
    )
    export_checkpoint(
        args.checkpoint,
        args.output_dir or default_output,
        args.validation_inputs,
        args.calibration_dir,
        args.int8_replay_dir,
    )


if __name__ == "__main__":
    main()
