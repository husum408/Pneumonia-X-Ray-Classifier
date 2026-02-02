from torch import nn

class CNN(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, kernel_size: int, padding: int, image_size_x: int, image_size_y: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=int(hidden_units),
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(hidden_units),
                      out_channels=int(hidden_units),
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=int(hidden_units)*int(image_size_x/8)*int(image_size_y/8),
                      out_features=output_shape)
        )

        self.features_conv = self.block_1[:14]

        self.gradients = None

        self.max_pool = nn.MaxPool2d(kernel_size=2,
                                     stride=2)

    # Hooks are later used to visualize activations

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        if x.requires_grad:
            x.register_hook(self.activations_hook)

        x = self.max_pool(x)
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)