# Environment setup note

Use conda to create env and use pip to install packages.

## PyTorch Nightly Install

need to be nightly

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade --force-reinstall
```

## Configure GCC and libstdc++ for compiling Triton locally

After installing nightly PyTorch, 
- remove the built-in triton by `pip uninstall pytorch-triton`
- check by `pip list | grep triton`
- build triton from source (not install nightly) to overwrite the old triton shipped with torch.

GCC module on Rivanna:

`gcc/11.4.0`: build error `collect2 ld exists with 1`

`gcc/13.3.0`: successfully installed 

Don't use BUILD_WITH_CLANG_LLD

Then install the latest `libstdc++` in the conda (should be at least 3.4.32)

```bash
conda install conda-forge::libstdcxx
```

Add them to the `LD_LIBRARY_PATH`
```bash
export LD_LIBRARY_PATH=/project/bi_dsc_large/fad3ew/.conda/gpt_fast/lib:$LD_LIBRARY_PATH
```

To check the `libstdc++` version
```bash
cd /project/bi_dsc_large/fad3ew/.conda/gpt_fast/lib
strings libstdc++.so.6 | grep GLIBCXX
```

## Full procedure

```bash
module load gcc/13.3.0 nccl/2.21.5-CUDA-12.4.1
conda activate gpt_fast
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade --force-reinstall
git clone https://github.com/triton-lang/triton.git
cd triton/python
pip install .
pip uninstall pytorch-triton
python -c "import torch; import triton; print(triton.__version__)"
```

or simply 
```bash
source compile_triton.sh
```

> sourcing shell scripts means running commands in current shell instead of subshells.

when need to checkout a remote branch
```bash
git fetch
git switch -c <local_branch> <upstream>/<branch>
```

## Library coode change

triton.compiler.CompiledKernel.launch_metadata

```python
def launch_metadata(self, grid, stream, *args):
    if CompiledKernel.launch_enter_hook is None:
        return None
    ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
    # if not isinstance(self.src, ASTSource) or self.src.fn.launch_metadata is None:
    if not isinstance(self.src, ASTSource): #TODO
        return ret
    arg_dict = {}
    arg_idx = 0
    # print(f"CompiledKernel launch_metadata src.fn.arg_names: {self.src.fn.arg_names}")
    # print(f"CompiledKernel launch_metadata src.constants: {self.src.constants}")
    for i, arg_name in enumerate(self.src.fn.arg_names):
        if i in self.src.fn.constexprs:
            # arg_dict[arg_name] = self.src.constants[arg_name] #BUG
            arg_dict[arg_name] = self.src.constants[arg_idx]
            arg_idx += 1
        else:
            arg_dict[arg_name] = args[arg_idx]
            arg_idx += 1
    # print(f"CompiledKernel launch_metadata produced arg_dict: {arg_dict}")
    ret.add(self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))
    return ret
```