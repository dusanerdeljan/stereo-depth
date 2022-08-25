#include "rescale_generated_view.hh"

namespace {
    template<typename scalar_t>
    __global__ void rescale_generated_view_kernel(
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output
    ) {
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t z = threadIdx.z;

        if ((x >= output.size(1)) || (y >= output.size(2))) {
            return;
        }

        scalar_t value = std::min<scalar_t>(std::max<scalar_t>(input[0][z][x][y] * 255 + 0.5, 0), 255);
        output[z][x][y] = value;
    }
};

void rescale_generated_view_cuda(
    torch::Tensor input,
    torch::Tensor output
) {
    const dim3 threads_per_block(16, 16, 3);
    const dim3 num_blocks(
        (input.size(2) + threads_per_block.x - 1) / threads_per_block.x,
        (input.size(3) + threads_per_block.y - 1) / threads_per_block.y,
        1
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "rescale_generated_view_cuda", ([&] {
        rescale_generated_view_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));
}