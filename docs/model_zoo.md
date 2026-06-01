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

The downloader supports both the current `hf download` command and the older
`huggingface-cli download` fallback.

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

The script prints the cloned GroundingDINO commit SHA. Save that SHA with the experiment log so
the detector code revision is traceable.

GroundingDINO also needs its BERT text encoder. By default `scripts/download_models.sh` pre-caches
these `bert-base-uncased` files in `HF_HOME`:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.txt`
- `model.safetensors`

It intentionally does not require `special_tokens_map.json`, because that file is not present for
all Hugging Face snapshots. Use `--skip-groundingdino-text-encoder` only if the cache is already
prepared or the host can download from Hugging Face during setup.

If the GitHub release download is slow from your cloud region, use the Hugging Face mirror/source
path:

```bash
bash scripts/download_models.sh --confirm --skip-llava --groundingdino-source hf
```

Then pass or configure:

```bash
--set verification.groundingdino.config_path=models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
--set verification.groundingdino.checkpoint_path=models/groundingdino_swint_ogc.pth
```

Install the package after the CUDA-compatible PyTorch stack is already installed:

```bash
python -m pip install -r requirements-models-cu12.txt
python -m pip install -U ninja wheel
python -m pip uninstall -y groundingdino || true
rm -rf models/GroundingDINO/build
find models/GroundingDINO/groundingdino -name '*.so' -type f -delete
python -m pip install -e models/GroundingDINO --no-build-isolation --no-deps
```

Rebuild the extension after changing torch or CUDA. Reusing an old `.so` after a torch upgrade is a
common source of import-time crashes.

## Reserved Methods

VCD and OPERA are intentionally not run by default. To add them later:

1. Install the upstream implementation in the `SVO` environment.
2. Add a model adapter that emits the same JSONL schema as `scripts/run_caption.py` or
   `scripts/run_pope.py`.
3. Register the method in `src/paper_reproduce/methods/registry.py`.
4. Add a method YAML under `configs/methods/`.
5. Include the method in `scripts/run_all.sh` only after the adapter produces real outputs.
