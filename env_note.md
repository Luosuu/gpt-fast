# Environment setup note

Use conda to create env and use pip to install packages.

## PyTorch

need to be nightly

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade --force-reinstall
```

## Triton

after installing nightly PyTorch, build triton from source (not install nightly) to overwrite the old triton shipped with torch.

remember to delete cache and recopy the source every time changing the compiler

problems:

`gcc/11.4.0`: build error `collect2 ld exists with 1`

`gcc/13.3.0`: successfully installed 

Don't use BUILD_WITH_CLANG_LLD

## Full procedure

```
module load gcc/13.3.0
conda activate gpt_fast
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade --force-reinstall
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -e python
python -c "import torch; import triton; print(triton.__version__)"
```