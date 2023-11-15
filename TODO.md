# General

- Add docs for all exported functions

# `nn` contents

- Store only modules and ops for them

## `nn.modules`

- Use glow.env as storage for options

# `zoo`

- Add `get_model() -> torch.nn.Module(stem=Stem(...), levels=[Level(...), ...], head=Head())`
- Add `VGG`, `ResNet`, `ResNet-D`, `ResNeXt`, `ResNeSt`, `Inception`, `DenseNet`, `EfficientNet`, `ViT`, `SWiN`
- Add `LinkNet`, `Unet`, `DeepLab`, `SkipNet`, `Tiramisu`, `MAnet`, `FPN`, `PAN`, `PSPNet`

# `optim` contents

- Subtype of `Iterable[float]` for lr policy.
- Class-adaptor for lr scheduler (batch/epoch-wise).
- Class-adaptor to combine optimizers.

# `data.get_loader`

- Seed as argument to toggle patching of dataset and iterable to provide batchsize- and workers-invariant data generation

# `util.plot_model` (from `plot`)

- Fix plotting to collapse standard modules, instead of falling through into them.
- Refactor visitor to be more readable.
