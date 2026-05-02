from flax import nnx


class ResNetBlock(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs):
        self.norm = nnx.GroupNorm(
            num_features=features,
            num_groups=None,
            group_size=1,
            rngs=rngs,
        )
        self.conv = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs,
        )

    def __call__(self, x):
        input_x = x
        x = nnx.relu(x)
        x = self.norm(x)
        x = self.conv(x)
        return input_x + x


class ImpalaBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.norm = nnx.GroupNorm(
            num_features=in_features,
            num_groups=None,
            group_size=1,
            rngs=rngs,
        )
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs,
        )
        self.resnet1 = ResNetBlock(
            features=out_features,
            rngs=rngs,
        )
        self.resnet2 = ResNetBlock(
            features=out_features,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = nnx.max_pool(x, (3, 3), (2, 2), "SAME")
        x = self.resnet1(x)
        x = self.resnet2(x)
        return x


class ImpalaCNN(nnx.Module):
    def __init__(self, channels: list[int], rngs: nnx.Rngs):
        self.blocks = []

        last_channel = 3
        for channel in channels:
            self.blocks.append(
                ImpalaBlock(
                    in_features=last_channel,
                    out_features=channel,
                    rngs=rngs,
                )
            )
            last_channel = channel

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x
