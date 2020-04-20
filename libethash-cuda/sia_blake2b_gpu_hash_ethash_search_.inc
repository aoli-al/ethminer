 __global__ __launch_bounds__(128, 3) void sia_blake2b_gpu_hash_ethash_search_fused_kernel_vfuse_lb_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, uint32_t iter4, volatile Search_results *g_output10, uint64_t start_nonce11)
 {
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
    for (int i = 0; i < iter4; i++) {
        uint32_t nonce5;
        nonce5 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter4 + i + startNonce1;
        static uint64_t s_target6 __attribute__((shared));
        if (!threadIdx_x_0)
            s_target6 = devectorize(target23);
        uint64_t m7[16];
        m7[0] = d_data2[0];
        m7[1] = d_data2[1];
        m7[2] = d_data2[2];
        m7[3] = d_data2[3];
        m7[4] = d_data2[4] | nonce5;
        m7[5] = d_data2[5];
        m7[6] = d_data2[6];
        m7[7] = d_data2[7];
        m7[8] = d_data2[8];
        m7[9] = d_data2[9];
        m7[10] = m7[11] = 0;
        m7[12] = m7[13] = m7[14] = m7[15] = 0;
        uint64_t v8[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(0, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(0, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(0, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(0, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(0, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(0, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(0, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(1, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(1, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(1, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(1, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(1, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(1, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(1, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(1, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(2, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(2, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(2, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(2, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(2, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(2, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(2, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(2, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(3, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(3, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(3, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(3, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(3, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(3, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(3, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(3, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(4, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(4, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(4, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(4, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(4, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(4, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(4, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(4, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(5, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(5, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(5, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(5, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(5, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(5, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(5, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(5, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(6, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(6, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(6, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(6, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(6, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(6, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(6, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(6, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(7, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(7, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(7, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(7, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(7, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(7, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(7, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(7, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(8, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(8, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(8, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(8, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(8, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(8, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(8, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(8, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(9, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(9, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(9, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(9, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(9, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(9, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(9, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(9, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(10, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(10, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(10, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(10, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(10, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(10, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(10, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(10, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(11, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(11, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(11, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(11, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(11, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(11, 5, v8[1], v8[6], v8[11], v8[12], m7);
        H(11, 6, v8[2], v8[7], v8[8], v8[13], m7);
        ;
        uint64_t h649;
        h649 = cuda_swab64(7640891576939301160L ^ v8[0] ^ v8[8]);
        if (h649 <= s_target6) {
            resNonce2[1] = resNonce2[0];
            resNonce2[0] = nonce5;
            s_target6 = h649;
        }
    }
}
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
    uint32_t gid12;
    gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14;
    nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15;
    mix_hash15 = mix13;
    bool result16;
    result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    int thread_id18;
    thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    int mix_idx19;
    mix_idx19 = thread_id18 & 3;
    for (int i = 0; i < (128 / 16); i += 4) {
        uint4 mix21[4];
        uint32_t offset22[4];
        uint32_t init023[4];
        for (int p = 0; p < 4; p++) {
            uint2 shuffle24[8];
            for (int j = 0; j < 8; j++) {
                shuffle24[j].x = __shfl_sync(4294967295U, (state17[j].x), (i + p), ((128 / 16)));
                shuffle24[j].y = __shfl_sync(4294967295U, (state17[j].y), (i + p), ((128 / 16)));
            }
            switch (mix_idx19) {
              case 0:
                mix21[p] = vectorize2(shuffle24[0], shuffle24[1]);
                break;
              case 1:
                mix21[p] = vectorize2(shuffle24[2], shuffle24[3]);
                break;
              case 2:
                mix21[p] = vectorize2(shuffle24[4], shuffle24[5]);
                break;
              case 3:
                mix21[p] = vectorize2(shuffle24[6], shuffle24[7]);
                break;
            }
            init023[p] = __shfl_sync(4294967295U, (shuffle24[0].x), (0), ((128 / 16)));
        }
        for (uint32_t a = 0; a < 64; a += 4) {
            int t25;
            t25 = bfe(a, 2U, 3U);
            for (uint32_t b = 0; b < 4; b++) {
                for (int p = 0; p < 4; p++) {
                    offset22[p] = ((init023[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix21[p])[b])) % d_dag_size;
                    offset22[p] = __shfl_sync(4294967295U, (offset22[p]), (t25), ((128 / 16)));
                    mix21[p] = fnv4(mix21[p], d_dag[offset22[p]].uint4s[thread_id18]);
                }
            }
        }
        for (int p = 0; p < 4; p++) {
            uint2 shuffle26[4];
            uint32_t thread_mix27;
            thread_mix27 = fnv_reduce(mix21[p]);
            shuffle26[0].x = __shfl_sync(4294967295U, (thread_mix27), (0), ((128 / 16)));
            shuffle26[0].y = __shfl_sync(4294967295U, (thread_mix27), (1), ((128 / 16)));
            shuffle26[1].x = __shfl_sync(4294967295U, (thread_mix27), (2), ((128 / 16)));
            shuffle26[1].y = __shfl_sync(4294967295U, (thread_mix27), (3), ((128 / 16)));
            shuffle26[2].x = __shfl_sync(4294967295U, (thread_mix27), (4), ((128 / 16)));
            shuffle26[2].y = __shfl_sync(4294967295U, (thread_mix27), (5), ((128 / 16)));
            shuffle26[3].x = __shfl_sync(4294967295U, (thread_mix27), (6), ((128 / 16)));
            shuffle26[3].y = __shfl_sync(4294967295U, (thread_mix27), (7), ((128 / 16)));
            if ((i + p) == thread_id18) {
                state17[8] = shuffle26[0];
                state17[9] = shuffle26[1];
                state17[10] = shuffle26[2];
                state17[11] = shuffle26[3];
            }
        }
    }
    if (!(cuda_swab64(keccak_f1600_final(state17)) > d_target)) {
        mix_hash15[0] = state17[8];
        mix_hash15[1] = state17[9];
        mix_hash15[2] = state17[10];
        mix_hash15[3] = state17[11];
        goto exit_point1;
    }
    uint32_t index20;
    index20 = atomicInc((uint32_t *)&g_output10->count, 4294967295U);
    if (index20 >= 4U)
        goto exit_point1;
    g_output10->result[index20].gid = gid12;
    g_output10->result[index20].mix[0] = mix13[0].x;
    g_output10->result[index20].mix[1] = mix13[0].y;
    g_output10->result[index20].mix[2] = mix13[1].x;
    g_output10->result[index20].mix[3] = mix13[1].y;
    g_output10->result[index20].mix[4] = mix13[2].x;
    g_output10->result[index20].mix[5] = mix13[2].y;
    g_output10->result[index20].mix[6] = mix13[3].x;
    g_output10->result[index20].mix[7] = mix13[3].y;
  exit_point1:
    ;
}
}
 __global__ __launch_bounds__(128, 0) void sia_blake2b_gpu_hash_ethash_search_fused_kernel_vfuse_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, uint32_t iter4, volatile Search_results *g_output10, uint64_t start_nonce11)
 {
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
    for (int i = 0; i < iter4; i++) {
        uint32_t nonce5;
        nonce5 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter4 + i + startNonce1;
        static uint64_t s_target6 __attribute__((shared));
        if (!threadIdx_x_0)
            s_target6 = devectorize(target23);
        uint64_t m7[16];
        m7[0] = d_data2[0];
        m7[1] = d_data2[1];
        m7[2] = d_data2[2];
        m7[3] = d_data2[3];
        m7[4] = d_data2[4] | nonce5;
        m7[5] = d_data2[5];
        m7[6] = d_data2[6];
        m7[7] = d_data2[7];
        m7[8] = d_data2[8];
        m7[9] = d_data2[9];
        m7[10] = m7[11] = 0;
        m7[12] = m7[13] = m7[14] = m7[15] = 0;
        uint64_t v8[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(0, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(0, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(0, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(0, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(0, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(0, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(0, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(1, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(1, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(1, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(1, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(1, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(1, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(1, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(1, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(2, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(2, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(2, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(2, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(2, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(2, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(2, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(2, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(3, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(3, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(3, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(3, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(3, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(3, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(3, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(3, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(4, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(4, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(4, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(4, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(4, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(4, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(4, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(4, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(5, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(5, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(5, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(5, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(5, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(5, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(5, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(5, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(6, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(6, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(6, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(6, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(6, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(6, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(6, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(6, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(7, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(7, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(7, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(7, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(7, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(7, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(7, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(7, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(8, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(8, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(8, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(8, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(8, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(8, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(8, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(8, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(9, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(9, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(9, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(9, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(9, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(9, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(9, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(9, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(10, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(10, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(10, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(10, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(10, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(10, 5, v8[1], v8[6], v8[11], v8[12], m7);
        G(10, 6, v8[2], v8[7], v8[8], v8[13], m7);
        G(10, 7, v8[3], v8[4], v8[9], v8[14], m7);
        ;
        G(11, 0, v8[0], v8[4], v8[8], v8[12], m7);
        G(11, 1, v8[1], v8[5], v8[9], v8[13], m7);
        G(11, 2, v8[2], v8[6], v8[10], v8[14], m7);
        G(11, 3, v8[3], v8[7], v8[11], v8[15], m7);
        G(11, 4, v8[0], v8[5], v8[10], v8[15], m7);
        G(11, 5, v8[1], v8[6], v8[11], v8[12], m7);
        H(11, 6, v8[2], v8[7], v8[8], v8[13], m7);
        ;
        uint64_t h649;
        h649 = cuda_swab64(7640891576939301160L ^ v8[0] ^ v8[8]);
        if (h649 <= s_target6) {
            resNonce2[1] = resNonce2[0];
            resNonce2[0] = nonce5;
            s_target6 = h649;
        }
    }
}
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
    uint32_t gid12;
    gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14;
    nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15;
    mix_hash15 = mix13;
    bool result16;
    result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    int thread_id18;
    thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    int mix_idx19;
    mix_idx19 = thread_id18 & 3;
    for (int i = 0; i < (128 / 16); i += 4) {
        uint4 mix21[4];
        uint32_t offset22[4];
        uint32_t init023[4];
        for (int p = 0; p < 4; p++) {
            uint2 shuffle24[8];
            for (int j = 0; j < 8; j++) {
                shuffle24[j].x = __shfl_sync(4294967295U, (state17[j].x), (i + p), ((128 / 16)));
                shuffle24[j].y = __shfl_sync(4294967295U, (state17[j].y), (i + p), ((128 / 16)));
            }
            switch (mix_idx19) {
              case 0:
                mix21[p] = vectorize2(shuffle24[0], shuffle24[1]);
                break;
              case 1:
                mix21[p] = vectorize2(shuffle24[2], shuffle24[3]);
                break;
              case 2:
                mix21[p] = vectorize2(shuffle24[4], shuffle24[5]);
                break;
              case 3:
                mix21[p] = vectorize2(shuffle24[6], shuffle24[7]);
                break;
            }
            init023[p] = __shfl_sync(4294967295U, (shuffle24[0].x), (0), ((128 / 16)));
        }
        for (uint32_t a = 0; a < 64; a += 4) {
            int t25;
            t25 = bfe(a, 2U, 3U);
            for (uint32_t b = 0; b < 4; b++) {
                for (int p = 0; p < 4; p++) {
                    offset22[p] = ((init023[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix21[p])[b])) % d_dag_size;
                    offset22[p] = __shfl_sync(4294967295U, (offset22[p]), (t25), ((128 / 16)));
                    mix21[p] = fnv4(mix21[p], d_dag[offset22[p]].uint4s[thread_id18]);
                }
            }
        }
        for (int p = 0; p < 4; p++) {
            uint2 shuffle26[4];
            uint32_t thread_mix27;
            thread_mix27 = fnv_reduce(mix21[p]);
            shuffle26[0].x = __shfl_sync(4294967295U, (thread_mix27), (0), ((128 / 16)));
            shuffle26[0].y = __shfl_sync(4294967295U, (thread_mix27), (1), ((128 / 16)));
            shuffle26[1].x = __shfl_sync(4294967295U, (thread_mix27), (2), ((128 / 16)));
            shuffle26[1].y = __shfl_sync(4294967295U, (thread_mix27), (3), ((128 / 16)));
            shuffle26[2].x = __shfl_sync(4294967295U, (thread_mix27), (4), ((128 / 16)));
            shuffle26[2].y = __shfl_sync(4294967295U, (thread_mix27), (5), ((128 / 16)));
            shuffle26[3].x = __shfl_sync(4294967295U, (thread_mix27), (6), ((128 / 16)));
            shuffle26[3].y = __shfl_sync(4294967295U, (thread_mix27), (7), ((128 / 16)));
            if ((i + p) == thread_id18) {
                state17[8] = shuffle26[0];
                state17[9] = shuffle26[1];
                state17[10] = shuffle26[2];
                state17[11] = shuffle26[3];
            }
        }
    }
    if (!(cuda_swab64(keccak_f1600_final(state17)) > d_target)) {
        mix_hash15[0] = state17[8];
        mix_hash15[1] = state17[9];
        mix_hash15[2] = state17[10];
        mix_hash15[3] = state17[11];
        goto exit_point1;
    }
    uint32_t index20;
    index20 = atomicInc((uint32_t *)&g_output10->count, 4294967295U);
    if (index20 >= 4U)
        goto exit_point1;
    g_output10->result[index20].gid = gid12;
    g_output10->result[index20].mix[0] = mix13[0].x;
    g_output10->result[index20].mix[1] = mix13[0].y;
    g_output10->result[index20].mix[2] = mix13[1].x;
    g_output10->result[index20].mix[3] = mix13[1].y;
    g_output10->result[index20].mix[4] = mix13[2].x;
    g_output10->result[index20].mix[5] = mix13[2].y;
    g_output10->result[index20].mix[6] = mix13[3].x;
    g_output10->result[index20].mix[7] = mix13[3].y;
  exit_point1:
    ;
}
}
 __global__ __launch_bounds__(256, 0) void sia_blake2b_gpu_hash_ethash_search_fused_kernel_hfuse_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, uint32_t iter4, volatile Search_results *g_output10, uint64_t start_nonce11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)) goto label_4;
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
for (int i = 0; i < iter4; i++) {
    uint32_t nonce5;
    nonce5 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter4 + i + startNonce1;
    static uint64_t s_target6 __attribute__((shared));
    if (!threadIdx_x_0)
        s_target6 = devectorize(target23);
    uint64_t m7[16];
    m7[0] = d_data2[0];
    m7[1] = d_data2[1];
    m7[2] = d_data2[2];
    m7[3] = d_data2[3];
    m7[4] = d_data2[4] | nonce5;
    m7[5] = d_data2[5];
    m7[6] = d_data2[6];
    m7[7] = d_data2[7];
    m7[8] = d_data2[8];
    m7[9] = d_data2[9];
    m7[10] = m7[11] = 0;
    m7[12] = m7[13] = m7[14] = m7[15] = 0;
    uint64_t v8[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
    G(0, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(0, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(0, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(0, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(0, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(0, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(0, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(0, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(1, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(1, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(1, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(1, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(1, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(1, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(1, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(1, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(2, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(2, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(2, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(2, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(2, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(2, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(2, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(2, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(3, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(3, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(3, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(3, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(3, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(3, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(3, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(3, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(4, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(4, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(4, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(4, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(4, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(4, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(4, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(4, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(5, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(5, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(5, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(5, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(5, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(5, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(5, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(5, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(6, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(6, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(6, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(6, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(6, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(6, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(6, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(6, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(7, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(7, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(7, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(7, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(7, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(7, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(7, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(7, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(8, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(8, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(8, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(8, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(8, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(8, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(8, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(8, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(9, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(9, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(9, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(9, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(9, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(9, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(9, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(9, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(10, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(10, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(10, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(10, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(10, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(10, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(10, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(10, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(11, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(11, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(11, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(11, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(11, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(11, 5, v8[1], v8[6], v8[11], v8[12], m7);
    H(11, 6, v8[2], v8[7], v8[8], v8[13], m7);
    ;
    uint64_t h649;
    h649 = cuda_swab64(7640891576939301160L ^ v8[0] ^ v8[8]);
    if (h649 <= s_target6) {
        resNonce2[1] = resNonce2[0];
        resNonce2[0] = nonce5;
        s_target6 = h649;
    }
}
label_4:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_5;
unsigned int blockDim_x_1;
blockDim_x_1 = 128;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
uint32_t gid12;
gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
uint2 mix13[4];
uint64_t nonce14;
nonce14 = start_nonce11 + gid12;
uint2 *mix_hash15;
mix_hash15 = mix13;
bool result16;
result16 = false;
uint2 state17[12];
state17[4] = vectorize(nonce14);
keccak_f1600_init(state17);
int thread_id18;
thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
int mix_idx19;
mix_idx19 = thread_id18 & 3;
for (int i = 0; i < (128 / 16); i += 4) {
    uint4 mix21[4];
    uint32_t offset22[4];
    uint32_t init023[4];
    for (int p = 0; p < 4; p++) {
        uint2 shuffle24[8];
        for (int j = 0; j < 8; j++) {
            shuffle24[j].x = __shfl_sync(4294967295U, (state17[j].x), (i + p), ((128 / 16)));
            shuffle24[j].y = __shfl_sync(4294967295U, (state17[j].y), (i + p), ((128 / 16)));
        }
        switch (mix_idx19) {
          case 0:
            mix21[p] = vectorize2(shuffle24[0], shuffle24[1]);
            break;
          case 1:
            mix21[p] = vectorize2(shuffle24[2], shuffle24[3]);
            break;
          case 2:
            mix21[p] = vectorize2(shuffle24[4], shuffle24[5]);
            break;
          case 3:
            mix21[p] = vectorize2(shuffle24[6], shuffle24[7]);
            break;
        }
        init023[p] = __shfl_sync(4294967295U, (shuffle24[0].x), (0), ((128 / 16)));
    }
    for (uint32_t a = 0; a < 64; a += 4) {
        int t25;
        t25 = bfe(a, 2U, 3U);
        for (uint32_t b = 0; b < 4; b++) {
            for (int p = 0; p < 4; p++) {
                offset22[p] = ((init023[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix21[p])[b])) % d_dag_size;
                offset22[p] = __shfl_sync(4294967295U, (offset22[p]), (t25), ((128 / 16)));
                mix21[p] = fnv4(mix21[p], d_dag[offset22[p]].uint4s[thread_id18]);
            }
        }
    }
    for (int p = 0; p < 4; p++) {
        uint2 shuffle26[4];
        uint32_t thread_mix27;
        thread_mix27 = fnv_reduce(mix21[p]);
        shuffle26[0].x = __shfl_sync(4294967295U, (thread_mix27), (0), ((128 / 16)));
        shuffle26[0].y = __shfl_sync(4294967295U, (thread_mix27), (1), ((128 / 16)));
        shuffle26[1].x = __shfl_sync(4294967295U, (thread_mix27), (2), ((128 / 16)));
        shuffle26[1].y = __shfl_sync(4294967295U, (thread_mix27), (3), ((128 / 16)));
        shuffle26[2].x = __shfl_sync(4294967295U, (thread_mix27), (4), ((128 / 16)));
        shuffle26[2].y = __shfl_sync(4294967295U, (thread_mix27), (5), ((128 / 16)));
        shuffle26[3].x = __shfl_sync(4294967295U, (thread_mix27), (6), ((128 / 16)));
        shuffle26[3].y = __shfl_sync(4294967295U, (thread_mix27), (7), ((128 / 16)));
        if ((i + p) == thread_id18) {
            state17[8] = shuffle26[0];
            state17[9] = shuffle26[1];
            state17[10] = shuffle26[2];
            state17[11] = shuffle26[3];
        }
    }
}
if (!(cuda_swab64(keccak_f1600_final(state17)) > d_target)) {
    mix_hash15[0] = state17[8];
    mix_hash15[1] = state17[9];
    mix_hash15[2] = state17[10];
    mix_hash15[3] = state17[11];
    goto exit_point1;
}
uint32_t index20;
index20 = atomicInc((uint32_t *)&g_output10->count, 4294967295U);
if (index20 >= 4U)
    goto exit_point1;
g_output10->result[index20].gid = gid12;
g_output10->result[index20].mix[0] = mix13[0].x;
g_output10->result[index20].mix[1] = mix13[0].y;
g_output10->result[index20].mix[2] = mix13[1].x;
g_output10->result[index20].mix[3] = mix13[1].y;
g_output10->result[index20].mix[4] = mix13[2].x;
g_output10->result[index20].mix[5] = mix13[2].y;
g_output10->result[index20].mix[6] = mix13[3].x;
g_output10->result[index20].mix[7] = mix13[3].y;
exit_point1:
;
label_5:;
}
 __global__ __launch_bounds__(256, 3) void sia_blake2b_gpu_hash_ethash_search_fused_kernel_hfuse_lb_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, uint32_t iter4, volatile Search_results *g_output10, uint64_t start_nonce11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)) goto label_6;
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
for (int i = 0; i < iter4; i++) {
    uint32_t nonce5;
    nonce5 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter4 + i + startNonce1;
    static uint64_t s_target6 __attribute__((shared));
    if (!threadIdx_x_0)
        s_target6 = devectorize(target23);
    uint64_t m7[16];
    m7[0] = d_data2[0];
    m7[1] = d_data2[1];
    m7[2] = d_data2[2];
    m7[3] = d_data2[3];
    m7[4] = d_data2[4] | nonce5;
    m7[5] = d_data2[5];
    m7[6] = d_data2[6];
    m7[7] = d_data2[7];
    m7[8] = d_data2[8];
    m7[9] = d_data2[9];
    m7[10] = m7[11] = 0;
    m7[12] = m7[13] = m7[14] = m7[15] = 0;
    uint64_t v8[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
    G(0, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(0, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(0, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(0, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(0, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(0, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(0, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(0, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(1, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(1, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(1, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(1, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(1, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(1, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(1, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(1, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(2, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(2, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(2, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(2, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(2, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(2, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(2, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(2, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(3, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(3, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(3, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(3, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(3, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(3, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(3, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(3, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(4, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(4, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(4, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(4, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(4, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(4, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(4, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(4, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(5, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(5, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(5, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(5, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(5, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(5, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(5, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(5, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(6, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(6, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(6, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(6, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(6, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(6, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(6, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(6, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(7, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(7, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(7, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(7, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(7, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(7, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(7, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(7, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(8, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(8, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(8, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(8, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(8, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(8, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(8, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(8, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(9, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(9, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(9, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(9, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(9, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(9, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(9, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(9, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(10, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(10, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(10, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(10, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(10, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(10, 5, v8[1], v8[6], v8[11], v8[12], m7);
    G(10, 6, v8[2], v8[7], v8[8], v8[13], m7);
    G(10, 7, v8[3], v8[4], v8[9], v8[14], m7);
    ;
    G(11, 0, v8[0], v8[4], v8[8], v8[12], m7);
    G(11, 1, v8[1], v8[5], v8[9], v8[13], m7);
    G(11, 2, v8[2], v8[6], v8[10], v8[14], m7);
    G(11, 3, v8[3], v8[7], v8[11], v8[15], m7);
    G(11, 4, v8[0], v8[5], v8[10], v8[15], m7);
    G(11, 5, v8[1], v8[6], v8[11], v8[12], m7);
    H(11, 6, v8[2], v8[7], v8[8], v8[13], m7);
    ;
    uint64_t h649;
    h649 = cuda_swab64(7640891576939301160L ^ v8[0] ^ v8[8]);
    if (h649 <= s_target6) {
        resNonce2[1] = resNonce2[0];
        resNonce2[0] = nonce5;
        s_target6 = h649;
    }
}
label_6:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_7;
unsigned int blockDim_x_1;
blockDim_x_1 = 128;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
uint32_t gid12;
gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
uint2 mix13[4];
uint64_t nonce14;
nonce14 = start_nonce11 + gid12;
uint2 *mix_hash15;
mix_hash15 = mix13;
bool result16;
result16 = false;
uint2 state17[12];
state17[4] = vectorize(nonce14);
keccak_f1600_init(state17);
int thread_id18;
thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
int mix_idx19;
mix_idx19 = thread_id18 & 3;
for (int i = 0; i < (128 / 16); i += 4) {
    uint4 mix21[4];
    uint32_t offset22[4];
    uint32_t init023[4];
    for (int p = 0; p < 4; p++) {
        uint2 shuffle24[8];
        for (int j = 0; j < 8; j++) {
            shuffle24[j].x = __shfl_sync(4294967295U, (state17[j].x), (i + p), ((128 / 16)));
            shuffle24[j].y = __shfl_sync(4294967295U, (state17[j].y), (i + p), ((128 / 16)));
        }
        switch (mix_idx19) {
          case 0:
            mix21[p] = vectorize2(shuffle24[0], shuffle24[1]);
            break;
          case 1:
            mix21[p] = vectorize2(shuffle24[2], shuffle24[3]);
            break;
          case 2:
            mix21[p] = vectorize2(shuffle24[4], shuffle24[5]);
            break;
          case 3:
            mix21[p] = vectorize2(shuffle24[6], shuffle24[7]);
            break;
        }
        init023[p] = __shfl_sync(4294967295U, (shuffle24[0].x), (0), ((128 / 16)));
    }
    for (uint32_t a = 0; a < 64; a += 4) {
        int t25;
        t25 = bfe(a, 2U, 3U);
        for (uint32_t b = 0; b < 4; b++) {
            for (int p = 0; p < 4; p++) {
                offset22[p] = ((init023[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix21[p])[b])) % d_dag_size;
                offset22[p] = __shfl_sync(4294967295U, (offset22[p]), (t25), ((128 / 16)));
                mix21[p] = fnv4(mix21[p], d_dag[offset22[p]].uint4s[thread_id18]);
            }
        }
    }
    for (int p = 0; p < 4; p++) {
        uint2 shuffle26[4];
        uint32_t thread_mix27;
        thread_mix27 = fnv_reduce(mix21[p]);
        shuffle26[0].x = __shfl_sync(4294967295U, (thread_mix27), (0), ((128 / 16)));
        shuffle26[0].y = __shfl_sync(4294967295U, (thread_mix27), (1), ((128 / 16)));
        shuffle26[1].x = __shfl_sync(4294967295U, (thread_mix27), (2), ((128 / 16)));
        shuffle26[1].y = __shfl_sync(4294967295U, (thread_mix27), (3), ((128 / 16)));
        shuffle26[2].x = __shfl_sync(4294967295U, (thread_mix27), (4), ((128 / 16)));
        shuffle26[2].y = __shfl_sync(4294967295U, (thread_mix27), (5), ((128 / 16)));
        shuffle26[3].x = __shfl_sync(4294967295U, (thread_mix27), (6), ((128 / 16)));
        shuffle26[3].y = __shfl_sync(4294967295U, (thread_mix27), (7), ((128 / 16)));
        if ((i + p) == thread_id18) {
            state17[8] = shuffle26[0];
            state17[9] = shuffle26[1];
            state17[10] = shuffle26[2];
            state17[11] = shuffle26[3];
        }
    }
}
if (!(cuda_swab64(keccak_f1600_final(state17)) > d_target)) {
    mix_hash15[0] = state17[8];
    mix_hash15[1] = state17[9];
    mix_hash15[2] = state17[10];
    mix_hash15[3] = state17[11];
    goto exit_point1;
}
uint32_t index20;
index20 = atomicInc((uint32_t *)&g_output10->count, 4294967295U);
if (index20 >= 4U)
    goto exit_point1;
g_output10->result[index20].gid = gid12;
g_output10->result[index20].mix[0] = mix13[0].x;
g_output10->result[index20].mix[1] = mix13[0].y;
g_output10->result[index20].mix[2] = mix13[1].x;
g_output10->result[index20].mix[3] = mix13[1].y;
g_output10->result[index20].mix[4] = mix13[2].x;
g_output10->result[index20].mix[5] = mix13[2].y;
g_output10->result[index20].mix[6] = mix13[3].x;
g_output10->result[index20].mix[7] = mix13[3].y;
exit_point1:
;
label_7:;
}
