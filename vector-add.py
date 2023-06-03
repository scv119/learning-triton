import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel1(
    look_up_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x_ptr = tl.load(look_up_ptr).to(tl.pointer_type(tl.float32))
    x = tl.load(x_ptr+ block_start + offsets, mask=mask)
    y = tl.load(y_ptr + block_start + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + block_start + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    output = torch.empty_like(x)

    n_elements = output.numel()

    #look_up_tensor = torch.tensor((x.data_ptr() - anchor.data_ptr()) / 4, dtype=torch.int64, device='cuda')
    look_up_tensor = torch.tensor(x.data_ptr(), dtype=torch.int64, device='cuda')

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    add_kernel1[grid](look_up_tensor, y, output, n_elements, BLOCK_SIZE=1024)

    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda', dtype=torch.float32)
y = torch.rand(size, device='cuda', dtype=torch.float32)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)