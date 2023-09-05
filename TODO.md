# General

- Add docs for all exported functions

### `__init__`

## `nn` contents

- Store only modules and ops for them

### `nn.modules`

- Use glow.env as storage for options

## `zoo`

- Add `get_model() -> torch.nn.Module(stem=Stem(...), levels=[Level(...), ...], head=Head())`
- Add `VGG`, `ResNet`, `ResNet-D`, `ResNeXt`, `ResNeSt`, `Inception`, `DenseNet`, `EfficientNet`, `ViT`, `SWiN`
- Add `LinkNet`, `Unet`, `DeepLab`, `SkipNet`, `Tiramisu`, `MAnet`, `FPN`, `PAN`, `PSPNet`

## `optim` contents (old `nn.optimizers`)

- Subtype of `Iterable[float]` for lr policy.
- Class-adaptor for lr scheduler (batch/epoch-wise).
- Dataclass for optimizer of single parameter group.
- Class-adaptor to combine optimizers.

## `util` contents

### `util.get_loader`

- Seed as argument to toggle patching of dataset and iterable to provide batchsize- and workers-invariant data generation

### `util.plot` (from `nn.plot`)

- Fix plotting to collapse standard modules, instead of falling through into them.
- Refactor visitor to be more readable.
