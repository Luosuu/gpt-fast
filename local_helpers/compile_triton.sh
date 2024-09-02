module load gcc/13.3.0 nccl/2.21.5-CUDA-12.4.1
conda activate gpt_fast
rm -rf /project/bi_dsc_large/fad3ew/triton
cd /project/bi_dsc_large/fad3ew/
git clone https://github.com/triton-lang/triton.git
export TRITON_HOME="/project/bi_dsc_large/fad3ew/triton"
which pip
which gcc
cd /project/bi_dsc_large/fad3ew/triton
pip install -e python
