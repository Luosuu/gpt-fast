```bash
python generate.py --compile --profile "profile/llama-2-7b" --use_proton --checkpoint_path "checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"
torchrun --standalone --nproc_per_node=4 generate.py --compile --profile "profile/llama-2-7b" --use_proton --checkpoint_path "checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"
```

```bash
proton-viewer -m time/s,tbyte/s profile/llama-2-7b_rank_None.hatchet -t 0.001
proton-viewer -m time/s,tbyte/s profile/llama-2-7b_rank_0.hatchet -t 0.001
```

```bash
python generate.py --compile --checkpoint_path "checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"
torchrun --standalone --nproc_per_node=4 generate.py --compile --checkpoint_path "checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"
```