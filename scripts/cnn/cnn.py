"""Five-block temporal CNN for windowed biosignals (Atzori et al., 2016).

Window tensors have shape ``(batch, channels, time)``. Sequence tensors have
shape ``(batch, sequence, channels, time)``. Each window is treated as a
single-map image with time as height and channels as width. The optional GRU
combines consecutive window embeddings before classification.
"""

import torch
from torch import nn


class TCNN(nn.Module):
    """Configurable five-block temporal convolutional classifier."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        time_kernel: int = 7,
        temporal_strides: tuple[int, int, int] = (2, 2, 1),
        temporal_dilations: tuple[int, int, int] = (1, 2, 4),
        n_filters: tuple[int, int, int, int] = (32, 32, 64, 64),
        dropout: float = 0.0,
        sequence_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_classes <= 0:
            raise ValueError("n_channels and n_classes must be positive")
        if time_kernel <= 0 or time_kernel % 2 == 0:
            raise ValueError("time_kernel must be a positive odd integer")
        if len(temporal_strides) != 3 or any(value <= 0 for value in temporal_strides):
            raise ValueError("temporal_strides must contain three positive integers")
        if len(temporal_dilations) != 3 or any(
            value <= 0 for value in temporal_dilations
        ):
            raise ValueError("temporal_dilations must contain three positive integers")
        if len(n_filters) != 4 or any(value <= 0 for value in n_filters):
            raise ValueError("n_filters must contain four positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if sequence_hidden_size is not None and sequence_hidden_size <= 0:
            raise ValueError("sequence_hidden_size must be positive or None")
        f1, f2, f3, f4 = n_filters
        s2, s3, s4 = temporal_strides
        d2, d3, d4 = temporal_dilations

        def temporal_convolution(
            in_channels: int,
            out_channels: int,
            *,
            stride: int,
            dilation: int,
        ) -> nn.Conv2d:
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(time_kernel, 1),
                stride=(stride, 1),
                padding=(dilation * (time_kernel - 1) // 2, 0),
                dilation=(dilation, 1),
            )

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, n_channels)),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            temporal_convolution(f1, f2, stride=s2, dilation=d2),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            temporal_convolution(f2, f3, stride=s3, dilation=d3),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            temporal_convolution(f3, f4, stride=s4, dilation=d4),
            nn.ReLU(),
        )
        self.feature_dropout = nn.Dropout2d(dropout)
        self.block5 = nn.Conv2d(f4, n_classes, kernel_size=1)
        self.sequence_gru = (
            None
            if sequence_hidden_size is None
            else nn.GRU(f4, sequence_hidden_size, batch_first=True)
        )
        self.sequence_classifier = (
            None
            if sequence_hidden_size is None
            else nn.Linear(sequence_hidden_size, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for window or consecutive-window sequence input."""

        if self.sequence_gru is None:
            if x.ndim != 3:
                raise ValueError(
                    "Window TCNN expects input shaped (batch, channels, time)"
                )
            features = self._window_feature_maps(x)
            return self.block5(features).mean(dim=(2, 3))

        if x.ndim != 4:
            raise ValueError(
                "Sequence TCNN expects input shaped "
                "(batch, sequence, channels, time)"
            )
        batch_size, sequence_length, channel_count, sample_count = x.shape
        windows = x.reshape(
            batch_size * sequence_length,
            channel_count,
            sample_count,
        )
        feature_maps = self._window_feature_maps(windows)
        embeddings = feature_maps.mean(dim=(2, 3)).reshape(
            batch_size,
            sequence_length,
            -1,
        )
        _, hidden = self.sequence_gru(embeddings)
        if self.sequence_classifier is None:
            raise RuntimeError("Sequence classifier is not configured")
        return self.sequence_classifier(hidden[-1])

    def _window_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Encode independent windows into convolutional feature maps."""

        x = x.transpose(1, 2).unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.feature_dropout(x)
