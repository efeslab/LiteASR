import triton 
import triton.language as tl
import torch 

@triton.jit
def _index_select_single_row_kernel(
    in_ptr,         # *pointer* to the input mask (float32 or whatever dtype)
    out_ptr,        # *pointer* to the output slice
    offset_ptr,     # *pointer* to the single-element tensor that has the row index
    stride_in_cols, # the leading dimension stride of 'mask' for row-major
    n_cols,         # number of columns in 'mask'
    BLOCK_SIZE: tl.constexpr
):
    """
    Copies one row from [M, N] input to [N] output.
    offset_ptr holds the row index we want.
    """
    # Compute the column indices this program will handle
    col_idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE

    # Each thread checks whether it is within [0..n_cols)
    mask = col_idx < n_cols

    # Load the single row index (scalar) from offset_ptr
    row_offset = tl.load(offset_ptr)  # shape=[1]

    # Address calculation:
    # row_offset * stride_in_cols gives the start of the row in the input
    in_row_start = row_offset * stride_in_cols
    in_addr  = in_ptr + in_row_start + col_idx
    out_addr = out_ptr + col_idx

    # Load from input row and store to output
    val = tl.load(in_addr, mask=mask, other=0.0)
    tl.store(out_addr, val, mask=mask)

def triton_index_select_single_row(mask: torch.Tensor, out: torch.Tensor, offset: torch.Tensor):
    """
    mask:   2D tensor of shape [M, N], on CUDA device
    offset: 1D scalar tensor [1], on the same device, giving the row index
    Returns: 1D tensor of shape [N]
    """
    # Check shapes
    assert mask.dim() == 2, "mask must be [M, N]"
    assert offset.numel() == 1, "offset must be a single-element tensor"

    M, N = mask.shape
    # We'll create an output to hold the selected row
    # out = torch.empty((N,), dtype=mask.dtype, device=mask.device)

    # Typically for row-major mask, stride(0) == N.
    # But let's not assume that we always have a perfect contiguous dimension:
    stride_in_cols = mask.stride(0)

    # We choose a block size for columns
    BLOCK_SIZE = 1024
    # The kernel grid is how many blocks we need along the columns dimension
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    # Launch the kernel
    _index_select_single_row_kernel[grid](
        mask,                # in_ptr
        out,                 # out_ptr
        offset,              # offset_ptr
        stride_in_cols,      # stride_in_cols
        N,                   # n_cols
        BLOCK_SIZE=BLOCK_SIZE
    )
    return 

@triton.jit
def fill_kv_cache_kernel(
    KV_KEY_PTR,        # float32*
    KV_VALUE_PTR,      # float32*
    X_KEY_PTR,         # float32*
    X_VALUE_PTR,       # float32*
    OFFSET_PTR,        # int32* (1-element)
    B: tl.constexpr,   # batch size
    S: tl.constexpr,   # sequence length
    H: tl.constexpr,   # hidden dimension
    BLOCK_H: tl.constexpr
):
    """
    Kernel that does:
       for b in [0..B):
         for h in [0..H):
             offset = *OFFSET_PTR
             kv_cache_key[b, offset, h]   = x_key[b, 0, h]
             kv_cache_value[b, offset, h] = x_value[b, 0, h]
    but in parallel blocks of size BLOCK_H along dimension H.
    """

    # Which batch index is this block handling?
    b_id = tl.program_id(0)
    # Which block along the hidden dimension?
    h_block_id = tl.program_id(1)

    # Compute the starting index in H for this block
    h_start = h_block_id * BLOCK_H
    # Compute a range [h_start, h_start+1, ... h_start + BLOCK_H - 1]
    h_range = h_start + tl.arange(0, BLOCK_H)

    # Load the offset from OFFSET_PTR (1-element tensor)
    offset = tl.load(OFFSET_PTR)

    # We only want to update indices h < H
    h_mask = h_range < H

    #--------------------------------------------------------------------------
    # 1) Update kv_cache_key:
    #    kv_cache_key[b_id, offset, h_range] = x_key[b_id, 0, h_range]
    # Pointer arithmetic for kv_cache_key:
    #   - Leading dimension is S*H for each batch,
    #   - then offset * H for the offset in [0..S),
    #   - then h_range for the hidden dimension.
    base_kv_key = (b_id * S + offset) * H
    kv_key_ptr_block = KV_KEY_PTR + base_kv_key + h_range

    # Pointer arithmetic for x_key:
    #   - shape [B, 1, H],
    #   - for dimension: (b_id * 1 + 0) * H + h_range
    base_x_key = (b_id * 1 + 0) * H
    x_key_ptr_block = X_KEY_PTR + base_x_key + h_range

    # Load from x_key
    x_key_vals = tl.load(x_key_ptr_block, mask=h_mask, other=0.0)
    # Store into kv_cache_key
    tl.store(kv_key_ptr_block, x_key_vals, mask=h_mask)

    #--------------------------------------------------------------------------
    # 2) Update kv_cache_value:
    #    kv_cache_value[b_id, offset, h_range] = x_value[b_id, 0, h_range]
    base_kv_value = (b_id * S + offset) * H
    kv_value_ptr_block = KV_VALUE_PTR + base_kv_value + h_range

    base_x_value = (b_id * 1 + 0) * H
    x_value_ptr_block = X_VALUE_PTR + base_x_value + h_range

    x_value_vals = tl.load(x_value_ptr_block, mask=h_mask, other=0.0)
    tl.store(kv_value_ptr_block, x_value_vals, mask=h_mask)

def fill_kv_cache_triton(kv_cache_key, kv_cache_value, x_key, x_value, offset_tensor):
    """
    Launch the Triton kernel fill_kv_cache_kernel() to do:
       kv_cache_key[:, offset, :]   = x_key[:, 0, :]
       kv_cache_value[:, offset, :] = x_value[:, 0, :]
    in parallel.

    kv_cache_key, kv_cache_value: shape [B, S, H]
    x_key, x_value:               shape [B, 1, H]
    offset_tensor:                shape [1], int32
    """
    assert kv_cache_key.is_cuda and kv_cache_value.is_cuda
    assert x_key.is_cuda and x_value.is_cuda
    assert offset_tensor.is_cuda
    assert offset_tensor.numel() == 1
    
    B, S, H = kv_cache_key.shape  # e.g., [1, 448, 1280]
    assert x_key.shape == (B, 1, H)
    assert x_value.shape == (B, 1, H)

    # Decide how big of a block in the hidden dimension.
    # Typically you might tune BLOCK_H for your device / hidden size.
    BLOCK_H = 128

    # Triton uses a 2D grid: 
    #   - dimension 0: B (batch blocks)
    #   - dimension 1: number of blocks covering H
    grid = (
        B,                          # program_id(0) -> b_id
        (H + BLOCK_H - 1) // BLOCK_H   # program_id(1) -> h_block_id
    )

    # Launch the kernel
    fill_kv_cache_kernel[grid](
        kv_cache_key,           # KV_KEY_PTR
        kv_cache_value,         # KV_VALUE_PTR
        x_key,                  # X_KEY_PTR
        x_value,                # X_VALUE_PTR
        offset_tensor,          # OFFSET_PTR
        B, S, H,                # shapes
        BLOCK_H=BLOCK_H,
        num_warps=4,            # Example tuning parameter
        num_stages=2            # Example tuning parameter
    )