_FSDP_USE_FULL_PREC_IN_EVAL=1 \
TORCH_LOGS="+dynamo" \
TORCHDYNAMO_VERBOSE=1 \
ENABLE_INTRA_NODE_COMM=1 \
torchrun --standalone --nproc_per_node=4 generate.py --compile \
    --profile "./profile/" \
    --checkpoint_path "checkpoints/meta-llama/Llama-2-7b-chat-hf/model_int8.pth"