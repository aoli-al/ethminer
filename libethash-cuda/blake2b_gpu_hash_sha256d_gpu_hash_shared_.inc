 __global__ __launch_bounds__(128, 8) void blake2b_gpu_hash_sha256d_gpu_hash_shared_fused_kernel_vfuse_lb_idx_0(const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1;
    blockDim_x_1 = 128;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter14; i++) {
        uint32_t nonce15;
        nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        uint64_t m16[16];
        m16[0] = d_blake_data[0];
        m16[1] = d_blake_data[1];
        m16[2] = d_blake_data[2];
        m16[3] = d_blake_data[3];
        m16[4] = d_blake_data[4];
        m16[5] = d_blake_data[5];
        m16[6] = d_blake_data[6];
        m16[7] = d_blake_data[7];
        m16[8] = d_blake_data[8];
        ((uint32_t *)m16)[18] = *((uint32_t *)(&d_blake_data[9]));
        ((uint32_t *)m16)[19] = nonce15;
        m16[10] = m16[11] = 0;
        m16[12] = m16[13] = 0;
        m16[14] = m16[15] = 0;
        uint64_t v17[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G_b(0, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(0, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(0, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(0, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(0, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(0, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(0, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(0, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(1, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(1, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(1, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(1, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(1, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(1, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(1, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(1, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(2, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(2, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(2, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(2, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(2, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(2, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(2, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(2, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(3, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(3, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(3, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(3, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(3, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(3, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(3, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(3, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(4, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(4, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(4, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(4, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(4, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(4, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(4, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(4, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(5, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(5, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(5, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(5, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(5, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(5, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(5, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(5, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(6, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(6, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(6, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(6, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(6, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(6, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(6, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(6, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(7, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(7, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(7, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(7, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(7, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(7, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(7, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(7, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(8, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(8, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(8, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(8, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(8, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(8, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(8, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(8, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(9, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(9, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(9, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(9, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(9, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(9, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(9, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(9, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(10, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(10, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(10, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(10, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(10, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(10, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(10, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(10, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(11, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(11, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(11, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(11, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(11, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(11, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(11, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(11, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        uint2 last18;
        last18 = vectorize(v17[3] ^ v17[11] ^ 11912009170470909681UL);
        if (last18.y <= target213.y && last18.x <= target213.x) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 128;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter3; i++) {
        uint32_t thread4;
        thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            uint32_t nonce6;
            nonce6 = startNonce1 + thread4;
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
            uint64_t high9;
            high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
}
 __global__ __launch_bounds__(128, 0) void blake2b_gpu_hash_sha256d_gpu_hash_shared_fused_kernel_vfuse_idx_0(const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1;
    blockDim_x_1 = 128;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter14; i++) {
        uint32_t nonce15;
        nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
        uint64_t m16[16];
        m16[0] = d_blake_data[0];
        m16[1] = d_blake_data[1];
        m16[2] = d_blake_data[2];
        m16[3] = d_blake_data[3];
        m16[4] = d_blake_data[4];
        m16[5] = d_blake_data[5];
        m16[6] = d_blake_data[6];
        m16[7] = d_blake_data[7];
        m16[8] = d_blake_data[8];
        ((uint32_t *)m16)[18] = *((uint32_t *)(&d_blake_data[9]));
        ((uint32_t *)m16)[19] = nonce15;
        m16[10] = m16[11] = 0;
        m16[12] = m16[13] = 0;
        m16[14] = m16[15] = 0;
        uint64_t v17[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G_b(0, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(0, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(0, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(0, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(0, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(0, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(0, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(0, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(1, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(1, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(1, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(1, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(1, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(1, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(1, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(1, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(2, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(2, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(2, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(2, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(2, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(2, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(2, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(2, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(3, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(3, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(3, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(3, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(3, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(3, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(3, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(3, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(4, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(4, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(4, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(4, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(4, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(4, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(4, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(4, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(5, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(5, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(5, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(5, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(5, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(5, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(5, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(5, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(6, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(6, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(6, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(6, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(6, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(6, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(6, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(6, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(7, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(7, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(7, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(7, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(7, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(7, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(7, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(7, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(8, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(8, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(8, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(8, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(8, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(8, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(8, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(8, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(9, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(9, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(9, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(9, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(9, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(9, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(9, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(9, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(10, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(10, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(10, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(10, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(10, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(10, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(10, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(10, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        G_b(11, 0, v17[0], v17[4], v17[8], v17[12], m16);
        G_b(11, 1, v17[1], v17[5], v17[9], v17[13], m16);
        G_b(11, 2, v17[2], v17[6], v17[10], v17[14], m16);
        G_b(11, 3, v17[3], v17[7], v17[11], v17[15], m16);
        G_b(11, 4, v17[0], v17[5], v17[10], v17[15], m16);
        G_b(11, 5, v17[1], v17[6], v17[11], v17[12], m16);
        G_b(11, 6, v17[2], v17[7], v17[8], v17[13], m16);
        G_b(11, 7, v17[3], v17[4], v17[9], v17[14], m16);
        ;
        uint2 last18;
        last18 = vectorize(v17[3] ^ v17[11] ^ 11912009170470909681UL);
        if (last18.y <= target213.y && last18.x <= target213.x) {
            resNonce12[1] = resNonce12[0];
            resNonce12[0] = nonce15;
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 128;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < iter3; i++) {
        uint32_t thread4;
        thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
        static uint32_t s_K5[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread4 < threads0) {
            uint32_t nonce6;
            nonce6 = startNonce1 + thread4;
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
            uint64_t high9;
            high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
            if (high9 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce6);
            }
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void blake2b_gpu_hash_sha256d_gpu_hash_shared_fused_kernel_hfuse_idx_0(const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)) goto label_4;
unsigned int blockDim_x_1;
blockDim_x_1 = 128;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
for (int i = 0; i < iter14; i++) {
    uint32_t nonce15;
    nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
    uint64_t m16[16];
    m16[0] = d_blake_data[0];
    m16[1] = d_blake_data[1];
    m16[2] = d_blake_data[2];
    m16[3] = d_blake_data[3];
    m16[4] = d_blake_data[4];
    m16[5] = d_blake_data[5];
    m16[6] = d_blake_data[6];
    m16[7] = d_blake_data[7];
    m16[8] = d_blake_data[8];
    ((uint32_t *)m16)[18] = *((uint32_t *)(&d_blake_data[9]));
    ((uint32_t *)m16)[19] = nonce15;
    m16[10] = m16[11] = 0;
    m16[12] = m16[13] = 0;
    m16[14] = m16[15] = 0;
    uint64_t v17[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
    G_b(0, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(0, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(0, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(0, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(0, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(0, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(0, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(0, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(1, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(1, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(1, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(1, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(1, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(1, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(1, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(1, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(2, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(2, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(2, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(2, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(2, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(2, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(2, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(2, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(3, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(3, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(3, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(3, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(3, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(3, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(3, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(3, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(4, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(4, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(4, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(4, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(4, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(4, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(4, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(4, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(5, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(5, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(5, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(5, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(5, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(5, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(5, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(5, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(6, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(6, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(6, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(6, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(6, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(6, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(6, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(6, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(7, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(7, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(7, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(7, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(7, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(7, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(7, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(7, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(8, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(8, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(8, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(8, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(8, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(8, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(8, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(8, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(9, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(9, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(9, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(9, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(9, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(9, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(9, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(9, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(10, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(10, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(10, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(10, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(10, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(10, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(10, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(10, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(11, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(11, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(11, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(11, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(11, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(11, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(11, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(11, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    uint2 last18;
    last18 = vectorize(v17[3] ^ v17[11] ^ 11912009170470909681UL);
    if (last18.y <= target213.y && last18.x <= target213.x) {
        resNonce12[1] = resNonce12[0];
        resNonce12[0] = nonce15;
    }
}
label_4:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_5;
unsigned int blockDim_x_0;
blockDim_x_0 = 128;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
for (int i = 0; i < iter3; i++) {
    uint32_t thread4;
    thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
    static uint32_t s_K5[256] __attribute__((shared));
    if (threadIdx_x_0 < 64U)
        s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
    if (thread4 < threads0) {
        uint32_t nonce6;
        nonce6 = startNonce1 + thread4;
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
        uint64_t high9;
        high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
        if (high9 <= c_target[0]) {
            resNonces2[1] = atomicExch(resNonces2, nonce6);
        }
    }
}
label_5:;
}
 __global__ __launch_bounds__(256, 8) void blake2b_gpu_hash_sha256d_gpu_hash_shared_fused_kernel_hfuse_lb_idx_0(const uint32_t threads10, const uint32_t startNonce11, uint32_t *resNonce12, const uint2 target213, uint32_t iter14, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)) goto label_6;
unsigned int blockDim_x_1;
blockDim_x_1 = 128;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
for (int i = 0; i < iter14; i++) {
    uint32_t nonce15;
    nonce15 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * iter14 + i + startNonce11;
    uint64_t m16[16];
    m16[0] = d_blake_data[0];
    m16[1] = d_blake_data[1];
    m16[2] = d_blake_data[2];
    m16[3] = d_blake_data[3];
    m16[4] = d_blake_data[4];
    m16[5] = d_blake_data[5];
    m16[6] = d_blake_data[6];
    m16[7] = d_blake_data[7];
    m16[8] = d_blake_data[8];
    ((uint32_t *)m16)[18] = *((uint32_t *)(&d_blake_data[9]));
    ((uint32_t *)m16)[19] = nonce15;
    m16[10] = m16[11] = 0;
    m16[12] = m16[13] = 0;
    m16[14] = m16[15] = 0;
    uint64_t v17[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
    G_b(0, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(0, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(0, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(0, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(0, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(0, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(0, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(0, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(1, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(1, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(1, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(1, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(1, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(1, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(1, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(1, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(2, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(2, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(2, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(2, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(2, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(2, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(2, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(2, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(3, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(3, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(3, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(3, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(3, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(3, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(3, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(3, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(4, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(4, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(4, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(4, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(4, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(4, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(4, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(4, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(5, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(5, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(5, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(5, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(5, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(5, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(5, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(5, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(6, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(6, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(6, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(6, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(6, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(6, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(6, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(6, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(7, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(7, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(7, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(7, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(7, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(7, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(7, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(7, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(8, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(8, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(8, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(8, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(8, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(8, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(8, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(8, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(9, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(9, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(9, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(9, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(9, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(9, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(9, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(9, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(10, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(10, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(10, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(10, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(10, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(10, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(10, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(10, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    G_b(11, 0, v17[0], v17[4], v17[8], v17[12], m16);
    G_b(11, 1, v17[1], v17[5], v17[9], v17[13], m16);
    G_b(11, 2, v17[2], v17[6], v17[10], v17[14], m16);
    G_b(11, 3, v17[3], v17[7], v17[11], v17[15], m16);
    G_b(11, 4, v17[0], v17[5], v17[10], v17[15], m16);
    G_b(11, 5, v17[1], v17[6], v17[11], v17[12], m16);
    G_b(11, 6, v17[2], v17[7], v17[8], v17[13], m16);
    G_b(11, 7, v17[3], v17[4], v17[9], v17[14], m16);
    ;
    uint2 last18;
    last18 = vectorize(v17[3] ^ v17[11] ^ 11912009170470909681UL);
    if (last18.y <= target213.y && last18.x <= target213.x) {
        resNonce12[1] = resNonce12[0];
        resNonce12[0] = nonce15;
    }
}
label_6:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_7;
unsigned int blockDim_x_0;
blockDim_x_0 = 128;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
for (int i = 0; i < iter3; i++) {
    uint32_t thread4;
    thread4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter3 + i;
    static uint32_t s_K5[256] __attribute__((shared));
    if (threadIdx_x_0 < 64U)
        s_K5[threadIdx_x_0] = c_K[threadIdx_x_0];
    if (thread4 < threads0) {
        uint32_t nonce6;
        nonce6 = startNonce1 + thread4;
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
        uint64_t high9;
        high9 = cuda_swab32ll(((uint64_t *)buf8)[3]);
        if (high9 <= c_target[0]) {
            resNonces2[1] = atomicExch(resNonces2, nonce6);
        }
    }
}
label_7:;
}
