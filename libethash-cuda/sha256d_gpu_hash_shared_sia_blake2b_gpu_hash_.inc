 __global__ __launch_bounds__(128, 5) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_vfuse_lb_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(128, 0) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_vfuse_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 32)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=32 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 224;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 32) % 224;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 32) / 224 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 32) / 224;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_idx_1(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 64)){
    unsigned int blockDim_x_0 = 64;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 64;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 64 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 64;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=64 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 192;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 64) % 192;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 64) / 192 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 64) / 192;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_idx_2(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 96)){
    unsigned int blockDim_x_0 = 96;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 96;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 96 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 96;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=96 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 160;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 96) % 160;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 96) / 160 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 96) / 160;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_idx_3(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_idx_4(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)){
    unsigned int blockDim_x_0 = 160;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 160;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 160 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 160;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 96;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) % 96;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) / 96 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) / 96;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_idx_5(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)){
    unsigned int blockDim_x_0 = 192;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 64;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 64;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 64 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 64;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_idx_6(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 224)){
    unsigned int blockDim_x_0 = 224;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 224;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 224 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 224;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=224 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 32;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 224) % 32;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 224) / 32 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 224) / 32;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 4) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 32)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=32 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 224;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 32) % 224;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 32) / 224 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 32) / 224;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 5) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_1(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 64)){
    unsigned int blockDim_x_0 = 64;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 64;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 64 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 64;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=64 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 192;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 64) % 192;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 64) / 192 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 64) / 192;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 6) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_2(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 96)){
    unsigned int blockDim_x_0 = 96;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 96;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 96 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 96;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=96 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 160;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 96) % 160;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 96) / 160 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 96) / 160;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 5) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_3(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 4) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_4(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)){
    unsigned int blockDim_x_0 = 160;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 160;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 160 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 160;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 96;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) % 96;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) / 96 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) / 96;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 3) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_5(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)){
    unsigned int blockDim_x_0 = 192;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 64;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 64;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 64 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 64;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
 __global__ __launch_bounds__(256, 3) void sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_6(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 224)){
    unsigned int blockDim_x_0 = 224;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 224;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 224 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 224;
    for (int i = 0; i < iter3; i++) {
        const uint32_t thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            const uint32_t nonce6 = startNonce1 + thread4;
            uint32_t dat7[16];
            *((uint2 *)(dat7)) = *((uint2 *)(c_dataEnd80));
            dat7[2] = c_dataEnd80[2];
            dat7[3] = nonce6;
            dat7[4] = 2147483648U;
            dat7[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat7[i] = 0;
            uint32_t buf8[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf8[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat7, buf8, s_K5);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat7[i] = buf8[i];
            dat7[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat7[i] = 0;
            dat7[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf8[i] = c_H256[i];
            sha256_round_last(dat7, buf8, s_K5);
            uint64_t high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=224 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 32;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 224) % 32;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 224) / 32 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 224) / 32;
    for (int i = 0; i < iter14; i++) {
        const uint32_t nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        static uint64_t s_target16 __attribute__((shared));
        if (!threadIdx_x_1)
            s_target16 = devectorize(target213);
        uint64_t m17[16];
        m17[0] = d_data2[0];
        m17[1] = d_data2[1];
        m17[2] = d_data2[2];
        m17[3] = d_data2[3];
        m17[4] = d_data2[4] | nonce15;
        m17[5] = d_data2[5];
        m17[6] = d_data2[6];
        m17[7] = d_data2[7];
        m17[8] = d_data2[8];
        m17[9] = d_data2[9];
        m17[10] = m17[11] = 0;
        m17[12] = m17[13] = m17[14] = m17[15] = 0;
        uint64_t v18[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(0, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(0, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(0, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(0, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(0, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(0, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(0, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(1, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(1, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(1, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(1, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(1, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(1, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(1, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(1, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(2, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(2, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(2, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(2, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(2, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(2, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(2, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(2, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(3, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(3, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(3, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(3, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(3, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(3, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(3, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(3, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(4, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(4, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(4, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(4, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(4, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(4, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(4, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(4, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(5, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(5, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(5, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(5, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(5, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(5, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(5, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(5, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(6, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(6, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(6, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(6, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(6, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(6, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(6, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(6, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(7, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(7, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(7, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(7, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(7, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(7, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(7, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(7, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(8, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(8, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(8, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(8, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(8, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(8, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(8, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(8, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(9, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(9, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(9, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(9, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(9, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(9, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(9, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(9, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(10, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(10, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(10, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(10, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(10, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(10, 5, v18[1], v18[6], v18[11], v18[12], m17);
        G(10, 6, v18[2], v18[7], v18[8], v18[13], m17);
        G(10, 7, v18[3], v18[4], v18[9], v18[14], m17);
        ;
        G(11, 0, v18[0], v18[4], v18[8], v18[12], m17);
        G(11, 1, v18[1], v18[5], v18[9], v18[13], m17);
        G(11, 2, v18[2], v18[6], v18[10], v18[14], m17);
        G(11, 3, v18[3], v18[7], v18[11], v18[15], m17);
        G(11, 4, v18[0], v18[5], v18[10], v18[15], m17);
        G(11, 5, v18[1], v18[6], v18[11], v18[12], m17);
        H(11, 6, v18[2], v18[7], v18[8], v18[13], m17);
        ;
        uint64_t h6419 = cuda_swab64(7640891576939301160L ^ v18[0] ^ v18[8]);
        if (h6419 <= s_target16) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
            s_target16 = h6419;
        }
    }
}
}
