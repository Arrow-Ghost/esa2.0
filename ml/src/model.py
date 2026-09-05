import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    1D CNN + LSTM fall-detection model.

    Input:
        (batch, 6, 400)

    Output:
        (batch, 2)

    Classes:
        0 = normal
        1 = fall
    """

    def __init__(
        self,
        input_channels=6,
        num_classes=2,
        cnn_channels=64,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.3,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                input_channels,
                cnn_channels,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                cnn_channels,
                cnn_channels,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x):
        # Input:
        # (batch, channels, time)
        x = self.cnn(x)

        # CNN output:
        # (batch, channels, reduced_time)

        # LSTM expects:
        # (batch, time, features)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)

        # Use the final timestep.
        x = x[:, -1, :]

        return self.classifier(x)


if __name__ == "__main__":
    print("Testing CNN-LSTM")
    print("================")

    model = CNNLSTM()

    test_input = torch.randn(8, 6, 400)

    output = model(test_input)

    print("Input shape: ", tuple(test_input.shape))
    print("Output shape:", tuple(output.shape))
    print()

    print("Expected input:  (8, 6, 400)")
    print("Expected output: (8, 2)")

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("Trainable parameters:", parameters)
