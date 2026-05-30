# models

Place local model checkpoints or symlinks here. This directory is intentionally ignored except for this README.

Planned model entry points:

- LLaVA-1.5-7B for caption generation and POPE answers.
- GroundingDINO for open-vocabulary object verification.

Set the actual paths in `configs/default.yaml`.

GroundingDINO integration expects the official [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) package interface:

```python
from groundingdino.util.inference import load_model, load_image, predict
```

Recommended local layout:

```text
models/
  groundingdino/
    GroundingDINO_SwinT_OGC.py
    groundingdino_swint_ogc.pth
```

Then set:

```yaml
verification:
  groundingdino:
    config_path: models/groundingdino/GroundingDINO_SwinT_OGC.py
    checkpoint_path: models/groundingdino/groundingdino_swint_ogc.pth
```

Do not report verification results until the package, config, and checkpoint are actually installed and used by `scripts/verify_objects.py`.

For cloud setup, use:

```bash
bash scripts/download_models.sh --confirm --install-groundingdino
python scripts/check_assets.py
```

The script requires explicit confirmation because LLaVA and detector assets are large and may need
authentication or license review.
