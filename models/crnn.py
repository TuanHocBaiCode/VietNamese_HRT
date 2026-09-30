import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=None, dropout=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        if pool is not None:
            layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    """
    CRNN baseline:
        CNN -> Map-to-Sequence -> 2-layer BiLSTM -> Linear -> CTC

    Input:
        [B, 1, 64, 512]

    CNN output:
        [B, 256, 1, 128]

    Sequence:
        [B, 128, 256]

    Output logits:
        [B, 128, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            # 64x512 -> 32x256
            ConvBlock(1, 64, pool=(2, 2), dropout=dropout),

            # 32x256 -> 16x128
            ConvBlock(64, 128, pool=(2, 2), dropout=dropout),

            # Height only: 16x128 -> 8x128
            ConvBlock(128, 256, pool=(2, 1), dropout=dropout),

            # 8x128 -> 4x128
            ConvBlock(256, 256, pool=(2, 1), dropout=dropout),

            # 4x128 -> 2x128
            ConvBlock(256, 256, pool=(2, 1), dropout=dropout),

            # 2x128 -> 1x128
            ConvBlock(256, 256, pool=(2, 1), dropout=dropout),

            # 7th convolutional layer, keep 1x128
            ConvBlock(256, 256, pool=None, dropout=dropout),
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, x):
        x = self.cnn(x)                     # [B, 256, 1, 128]
        x = x.squeeze(2).permute(0, 2, 1)  # [B, 128, 256]
        x, _ = self.lstm(x)                # [B, 128, 512]
        logits = self.classifier(x)        # [B, 128, C]
        return logits

    @torch.no_grad()
    def output_lengths(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return torch.full(
            (batch_size,),
            self.cnn_output_width,
            dtype=torch.long,
            device=device,
        )

    @property
    def cnn_output_width(self):
        return 128


__all__ = ["CRNN"]
