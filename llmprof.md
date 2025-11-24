# README

## Python Env

```bash
git submodule init --update --recursive
uv sync --extra cuda
```

## Prepare models

Download

```bash
python scripts/download.py --repo_id meta-llama/Meta-Llama-3.1-8B --local_dir /mnt/local/localcache00/llama-3.1-8b --hf_token xxxx
```

Convert

```bash
python scripts/convert_hf_checkpoint.py --checkpoint_dir /mnt/local/localcache00/llama-3.1-8b/ --model_name llama-3.1-8b
```

## Run

```bash
python generate.py --compile --profile "profiles/llama-3.1-8b" --checkpoint_path /path/to/llama-3.1-8b/model.pth --model_name llama-3.1-8b --tokenizer_path /path/to/llama-3.1-8b/tokenizer.model --prompt 1000 --use_proton --profiler-hook triton
```

