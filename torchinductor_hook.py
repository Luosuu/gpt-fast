from triton.profiler.scope import enter_scope, exit_scope
from triton.compiler import CompiledKernel, LazyDict
import torch

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"

# original TritonHook
# class TritonHook:
#     flops_width = [8, 16, 32, 64]
#     metrics = [f"flops{width}" for width in flops_width] + ["bytes"] + ["flops"]

#     @staticmethod
#     def enter(lazy_dict: LazyDict) -> None:
#         enter_scope(COMPUTE_METADATA_SCOPE_NAME)
#         metadata = lazy_dict.get()
#         exit_scope()
#         fn_metrics = {k: metadata[k] for k in TritonHook.metrics if k in metadata}
#         enter_scope(metadata["name"], triton_op=True, metrics=fn_metrics)

#     @staticmethod
#     def exit(lazy_dict: LazyDict) -> None:
#         exit_scope(triton_op=True)

# def register_triton_hook() -> None:
#     if CompiledKernel.launch_enter_hook is None:
#         CompiledKernel.launch_enter_hook = enter
#         CompiledKernel.launch_exit_hook = exit


# def unregister_triton_hook() -> None:
#     if CompiledKernel.launch_enter_hook == enter:
#         CompiledKernel.launch_enter_hook = None
#         CompiledKernel.launch_exit_hook = None

def exit(lazy_dict: LazyDict) -> None:
    exit_scope(triton_op=True)

def enter(lazy_dict: LazyDict) -> None:
    enter_scope(COMPUTE_METADATA_SCOPE_NAME)
    # metadata = lazy_dict.get()
    # exit_scope()
    # fn_metrics = {k: metadata[k] for k in TritonHook.metrics if k in metadata}

    # print(f"torchinductor hook enter - lazy_dict.data: {lazy_dict.data}")
    # print(f"torchinductor hook enter - lazy_dict.extras: {lazy_dict.extras}")
    _, (_, kernel_metadata, extra_dict) = lazy_dict.extras[0]
    # print(f"kernel metadata: {kernel_metadata}")
    # print(f"extra_dict: {extra_dict}")

    num_warps = kernel_metadata.num_warps
    num_stages = kernel_metadata.num_stages
    cluster_x, cluster_y, cluster_z = kernel_metadata.cluster_dims
    shared_memory = kernel_metadata.shared

    # print(lazy_dict.data["name"])
    metadata = {
        'name': f"{lazy_dict.data['name']}_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>"
    }
    # print(metadata['name'])
    # for key in extra_dict:
    #     if torch.is_tensor(extra_dict[key]):
    #         print(f"{key} shape: {extra_dict[key].shape}")

    bytes = sum([torch.numel(extra_dict[key])*extra_dict[key].element_size() for key in extra_dict if torch.is_tensor(extra_dict[key])])
    exit_scope()
    print(metadata['name'])
    print(bytes)
    fn_metrics={'bytes': bytes}
    enter_scope(metadata['name'], triton_op=True, metrics=fn_metrics)