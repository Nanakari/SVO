# Model Zoo and Checkpoints

The repository stores only paths and download helpers. Do not commit model weights.

## LLaVA-1.5-7B

Default model id:

```text
llava-hf/llava-1.5-7b-hf
```

Download after installing `huggingface_hub`:

```bash
bash scripts/download_models.sh --confirm --skip-groundingdino
```

Use a local path by overriding:

```bash
--set generation.model_name_or_path=models/llava-1.5-7b-hf
```

## GroundingDINO

The verifier adapter imports the official package interface:

```python
from groundingdino.util.inference import load_model, load_image, predict
```

Download/clone assets:

```bash
bash scripts/download_models.sh --confirm --skip-llava --install-groundingdino
```

Then pass or configure:

```bash
--set verification.groundingdino.config_path=models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
--set verification.groundingdino.checkpoint_path=models/groundingdino_swint_ogc.pth
```

## Reserved Methods

VCD and OPERA are intentionally not run by default. To add them later:

1. Install the upstream implementation in the `SVO` environment.
2. Add a model adapter that emits the same JSONL schema as `scripts/run_caption.py` or
   `scripts/run_pope.py`.
3. Register the method in `src/paper_reproduce/methods/registry.py`.
4. Add a method YAML under `configs/methods/`.
5. Include the method in `scripts/run_all.sh` only after the adapter produces real outputs.
