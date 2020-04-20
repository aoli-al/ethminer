 __global__ __launch_bounds__(128, 3) void ethash_search_blake2b_gpu_hash_fused_kernel_vfuse_lb_idx_0(volatile Search_results *g_output9, uint64_t start_nonce10, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, uint32_t iter4)
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
    uint32_t gid11;
    gid11 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix12[4];
    uint64_t nonce13;
    nonce13 = start_nonce10 + gid11;
    uint2 *mix_hash14;
    mix_hash14 = mix12;
    bool result15;
    result15 = false;
    uint2 state16[12];
    state16[4] = vectorize(nonce13);
    keccak_f1600_init(state16);
    int thread_id17;
    thread_id17 = threadIdx_x_1 & ((128 / 16) - 1);
    int mix_idx18;
    mix_idx18 = thread_id17 & 3;
    for (int i = 0; i < (128 / 16); i += 4) {
        uint4 mix20[4];
        uint32_t offset21[4];
        uint32_t init022[4];
        for (int p = 0; p < 4; p++) {
            uint2 shuffle23[8];
            for (int j = 0; j < 8; j++) {
                shuffle23[j].x = __shfl_sync(4294967295U, (state16[j].x), (i + p), ((128 / 16)));
                shuffle23[j].y = __shfl_sync(4294967295U, (state16[j].y), (i + p), ((128 / 16)));
            }
            switch (mix_idx18) {
              case 0:
                mix20[p] = vectorize2(shuffle23[0], shuffle23[1]);
                break;
              case 1:
                mix20[p] = vectorize2(shuffle23[2], shuffle23[3]);
                break;
              case 2:
                mix20[p] = vectorize2(shuffle23[4], shuffle23[5]);
                break;
              case 3:
                mix20[p] = vectorize2(shuffle23[6], shuffle23[7]);
                break;
            }
            init022[p] = __shfl_sync(4294967295U, (shuffle23[0].x), (0), ((128 / 16)));
        }
        for (uint32_t a = 0; a < 64; a += 4) {
            int t24;
            t24 = bfe(a, 2U, 3U);
            for (uint32_t b = 0; b < 4; b++) {
                for (int p = 0; p < 4; p++) {
                    offset21[p] = ((init022[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix20[p])[b])) % d_dag_size;
                    offset21[p] = __shfl_sync(4294967295U, (offset21[p]), (t24), ((128 / 16)));
                    mix20[p] = fnv4(mix20[p], d_dag[offset21[p]].uint4s[thread_id17]);
                }
            }
        }
        for (int p = 0; p < 4; p++) {
            uint2 shuffle25[4];
            uint32_t thread_mix26;
            thread_mix26 = fnv_reduce(mix20[p]);
            shuffle25[0].x = __shfl_sync(4294967295U, (thread_mix26), (0), ((128 / 16)));
            shuffle25[0].y = __shfl_sync(4294967295U, (thread_mix26), (1), ((128 / 16)));
            shuffle25[1].x = __shfl_sync(4294967295U, (thread_mix26), (2), ((128 / 16)));
            shuffle25[1].y = __shfl_sync(4294967295U, (thread_mix26), (3), ((128 / 16)));
            shuffle25[2].x = __shfl_sync(4294967295U, (thread_mix26), (4), ((128 / 16)));
            shuffle25[2].y = __shfl_sync(4294967295U, (thread_mix26), (5), ((128 / 16)));
            shuffle25[3].x = __shfl_sync(4294967295U, (thread_mix26), (6), ((128 / 16)));
            shuffle25[3].y = __shfl_sync(4294967295U, (thread_mix26), (7), ((128 / 16)));
            if ((i + p) == thread_id17) {
                state16[8] = shuffle25[0];
                state16[9] = shuffle25[1];
                state16[10] = shuffle25[2];
                state16[11] = shuffle25[3];
            }
        }
    }
    if (!(cuda_swab64(keccak_f1600_final(state16)) > d_target)) {
        mix_hash14[0] = state16[8];
        mix_hash14[1] = state16[9];
        mix_hash14[2] = state16[10];
        mix_hash14[3] = state16[11];
        goto exit_point1;
    }
    uint32_t index19;
    index19 = atomicInc((uint32_t *)&g_output9->count, 4294967295U);
    if (index19 >= 4U)
        goto exit_point1;
    g_output9->result[index19].gid = gid11;
    g_output9->result[index19].mix[0] = mix12[0].x;
    g_output9->result[index19].mix[1] = mix12[0].y;
    g_output9->result[index19].mix[2] = mix12[1].x;
    g_output9->result[index19].mix[3] = mix12[1].y;
    g_output9->result[index19].mix[4] = mix12[2].x;
    g_output9->result[index19].mix[5] = mix12[2].y;
    g_output9->result[index19].mix[6] = mix12[3].x;
    g_output9->result[index19].mix[7] = mix12[3].y;
  exit_point1:
    ;
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
    for (int i = 0; i < iter4; i++) {
        uint32_t nonce5;
        nonce5 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter4 + i + startNonce1;
        uint64_t m6[16];
        m6[0] = d_blake_data[0];
        m6[1] = d_blake_data[1];
        m6[2] = d_blake_data[2];
        m6[3] = d_blake_data[3];
        m6[4] = d_blake_data[4];
        m6[5] = d_blake_data[5];
        m6[6] = d_blake_data[6];
        m6[7] = d_blake_data[7];
        m6[8] = d_blake_data[8];
        ((uint32_t *)m6)[18] = *((uint32_t *)(&d_blake_data[9]));
        ((uint32_t *)m6)[19] = nonce5;
        m6[10] = m6[11] = 0;
        m6[12] = m6[13] = 0;
        m6[14] = m6[15] = 0;
        uint64_t v7[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G_b(0, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(0, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(0, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(0, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(0, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(0, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(0, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(0, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(1, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(1, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(1, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(1, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(1, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(1, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(1, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(1, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(2, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(2, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(2, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(2, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(2, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(2, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(2, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(2, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(3, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(3, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(3, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(3, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(3, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(3, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(3, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(3, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(4, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(4, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(4, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(4, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(4, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(4, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(4, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(4, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(5, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(5, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(5, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(5, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(5, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(5, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(5, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(5, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(6, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(6, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(6, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(6, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(6, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(6, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(6, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(6, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(7, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(7, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(7, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(7, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(7, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(7, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(7, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(7, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(8, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(8, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(8, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(8, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(8, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(8, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(8, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(8, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(9, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(9, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(9, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(9, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(9, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(9, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(9, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(9, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(10, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(10, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(10, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(10, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(10, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(10, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(10, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(10, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(11, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(11, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(11, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(11, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(11, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(11, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(11, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(11, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        uint2 last8;
        last8 = vectorize(v7[3] ^ v7[11] ^ 11912009170470909681UL);
        if (last8.y <= target23.y && last8.x <= target23.x) {
            resNonce2[1] = resNonce2[0];
            resNonce2[0] = nonce5;
        }
    }
}
}
 __global__ __launch_bounds__(128, 0) void ethash_search_blake2b_gpu_hash_fused_kernel_vfuse_idx_0(volatile Search_results *g_output9, uint64_t start_nonce10, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, uint32_t iter4)
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
    uint32_t gid11;
    gid11 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix12[4];
    uint64_t nonce13;
    nonce13 = start_nonce10 + gid11;
    uint2 *mix_hash14;
    mix_hash14 = mix12;
    bool result15;
    result15 = false;
    uint2 state16[12];
    state16[4] = vectorize(nonce13);
    keccak_f1600_init(state16);
    int thread_id17;
    thread_id17 = threadIdx_x_1 & ((128 / 16) - 1);
    int mix_idx18;
    mix_idx18 = thread_id17 & 3;
    for (int i = 0; i < (128 / 16); i += 4) {
        uint4 mix20[4];
        uint32_t offset21[4];
        uint32_t init022[4];
        for (int p = 0; p < 4; p++) {
            uint2 shuffle23[8];
            for (int j = 0; j < 8; j++) {
                shuffle23[j].x = __shfl_sync(4294967295U, (state16[j].x), (i + p), ((128 / 16)));
                shuffle23[j].y = __shfl_sync(4294967295U, (state16[j].y), (i + p), ((128 / 16)));
            }
            switch (mix_idx18) {
              case 0:
                mix20[p] = vectorize2(shuffle23[0], shuffle23[1]);
                break;
              case 1:
                mix20[p] = vectorize2(shuffle23[2], shuffle23[3]);
                break;
              case 2:
                mix20[p] = vectorize2(shuffle23[4], shuffle23[5]);
                break;
              case 3:
                mix20[p] = vectorize2(shuffle23[6], shuffle23[7]);
                break;
            }
            init022[p] = __shfl_sync(4294967295U, (shuffle23[0].x), (0), ((128 / 16)));
        }
        for (uint32_t a = 0; a < 64; a += 4) {
            int t24;
            t24 = bfe(a, 2U, 3U);
            for (uint32_t b = 0; b < 4; b++) {
                for (int p = 0; p < 4; p++) {
                    offset21[p] = ((init022[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix20[p])[b])) % d_dag_size;
                    offset21[p] = __shfl_sync(4294967295U, (offset21[p]), (t24), ((128 / 16)));
                    mix20[p] = fnv4(mix20[p], d_dag[offset21[p]].uint4s[thread_id17]);
                }
            }
        }
        for (int p = 0; p < 4; p++) {
            uint2 shuffle25[4];
            uint32_t thread_mix26;
            thread_mix26 = fnv_reduce(mix20[p]);
            shuffle25[0].x = __shfl_sync(4294967295U, (thread_mix26), (0), ((128 / 16)));
            shuffle25[0].y = __shfl_sync(4294967295U, (thread_mix26), (1), ((128 / 16)));
            shuffle25[1].x = __shfl_sync(4294967295U, (thread_mix26), (2), ((128 / 16)));
            shuffle25[1].y = __shfl_sync(4294967295U, (thread_mix26), (3), ((128 / 16)));
            shuffle25[2].x = __shfl_sync(4294967295U, (thread_mix26), (4), ((128 / 16)));
            shuffle25[2].y = __shfl_sync(4294967295U, (thread_mix26), (5), ((128 / 16)));
            shuffle25[3].x = __shfl_sync(4294967295U, (thread_mix26), (6), ((128 / 16)));
            shuffle25[3].y = __shfl_sync(4294967295U, (thread_mix26), (7), ((128 / 16)));
            if ((i + p) == thread_id17) {
                state16[8] = shuffle25[0];
                state16[9] = shuffle25[1];
                state16[10] = shuffle25[2];
                state16[11] = shuffle25[3];
            }
        }
    }
    if (!(cuda_swab64(keccak_f1600_final(state16)) > d_target)) {
        mix_hash14[0] = state16[8];
        mix_hash14[1] = state16[9];
        mix_hash14[2] = state16[10];
        mix_hash14[3] = state16[11];
        goto exit_point1;
    }
    uint32_t index19;
    index19 = atomicInc((uint32_t *)&g_output9->count, 4294967295U);
    if (index19 >= 4U)
        goto exit_point1;
    g_output9->result[index19].gid = gid11;
    g_output9->result[index19].mix[0] = mix12[0].x;
    g_output9->result[index19].mix[1] = mix12[0].y;
    g_output9->result[index19].mix[2] = mix12[1].x;
    g_output9->result[index19].mix[3] = mix12[1].y;
    g_output9->result[index19].mix[4] = mix12[2].x;
    g_output9->result[index19].mix[5] = mix12[2].y;
    g_output9->result[index19].mix[6] = mix12[3].x;
    g_output9->result[index19].mix[7] = mix12[3].y;
  exit_point1:
    ;
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
    for (int i = 0; i < iter4; i++) {
        uint32_t nonce5;
        nonce5 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter4 + i + startNonce1;
        uint64_t m6[16];
        m6[0] = d_blake_data[0];
        m6[1] = d_blake_data[1];
        m6[2] = d_blake_data[2];
        m6[3] = d_blake_data[3];
        m6[4] = d_blake_data[4];
        m6[5] = d_blake_data[5];
        m6[6] = d_blake_data[6];
        m6[7] = d_blake_data[7];
        m6[8] = d_blake_data[8];
        ((uint32_t *)m6)[18] = *((uint32_t *)(&d_blake_data[9]));
        ((uint32_t *)m6)[19] = nonce5;
        m6[10] = m6[11] = 0;
        m6[12] = m6[13] = 0;
        m6[14] = m6[15] = 0;
        uint64_t v7[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G_b(0, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(0, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(0, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(0, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(0, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(0, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(0, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(0, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(1, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(1, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(1, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(1, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(1, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(1, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(1, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(1, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(2, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(2, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(2, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(2, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(2, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(2, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(2, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(2, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(3, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(3, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(3, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(3, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(3, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(3, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(3, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(3, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(4, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(4, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(4, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(4, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(4, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(4, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(4, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(4, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(5, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(5, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(5, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(5, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(5, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(5, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(5, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(5, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(6, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(6, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(6, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(6, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(6, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(6, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(6, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(6, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(7, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(7, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(7, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(7, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(7, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(7, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(7, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(7, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(8, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(8, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(8, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(8, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(8, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(8, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(8, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(8, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(9, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(9, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(9, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(9, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(9, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(9, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(9, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(9, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(10, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(10, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(10, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(10, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(10, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(10, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(10, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(10, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G_b(11, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G_b(11, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G_b(11, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G_b(11, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G_b(11, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G_b(11, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G_b(11, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G_b(11, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        uint2 last8;
        last8 = vectorize(v7[3] ^ v7[11] ^ 11912009170470909681UL);
        if (last8.y <= target23.y && last8.x <= target23.x) {
            resNonce2[1] = resNonce2[0];
            resNonce2[0] = nonce5;
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void ethash_search_blake2b_gpu_hash_fused_kernel_hfuse_idx_0(volatile Search_results *g_output9, uint64_t start_nonce10, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, uint32_t iter4)
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
uint32_t gid11;
gid11 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
uint2 mix12[4];
uint64_t nonce13;
nonce13 = start_nonce10 + gid11;
uint2 *mix_hash14;
mix_hash14 = mix12;
bool result15;
result15 = false;
uint2 state16[12];
state16[4] = vectorize(nonce13);
keccak_f1600_init(state16);
int thread_id17;
thread_id17 = threadIdx_x_1 & ((128 / 16) - 1);
int mix_idx18;
mix_idx18 = thread_id17 & 3;
for (int i = 0; i < (128 / 16); i += 4) {
    uint4 mix20[4];
    uint32_t offset21[4];
    uint32_t init022[4];
    for (int p = 0; p < 4; p++) {
        uint2 shuffle23[8];
        for (int j = 0; j < 8; j++) {
            shuffle23[j].x = __shfl_sync(4294967295U, (state16[j].x), (i + p), ((128 / 16)));
            shuffle23[j].y = __shfl_sync(4294967295U, (state16[j].y), (i + p), ((128 / 16)));
        }
        switch (mix_idx18) {
          case 0:
            mix20[p] = vectorize2(shuffle23[0], shuffle23[1]);
            break;
          case 1:
            mix20[p] = vectorize2(shuffle23[2], shuffle23[3]);
            break;
          case 2:
            mix20[p] = vectorize2(shuffle23[4], shuffle23[5]);
            break;
          case 3:
            mix20[p] = vectorize2(shuffle23[6], shuffle23[7]);
            break;
        }
        init022[p] = __shfl_sync(4294967295U, (shuffle23[0].x), (0), ((128 / 16)));
    }
    for (uint32_t a = 0; a < 64; a += 4) {
        int t24;
        t24 = bfe(a, 2U, 3U);
        for (uint32_t b = 0; b < 4; b++) {
            for (int p = 0; p < 4; p++) {
                offset21[p] = ((init022[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix20[p])[b])) % d_dag_size;
                offset21[p] = __shfl_sync(4294967295U, (offset21[p]), (t24), ((128 / 16)));
                mix20[p] = fnv4(mix20[p], d_dag[offset21[p]].uint4s[thread_id17]);
            }
        }
    }
    for (int p = 0; p < 4; p++) {
        uint2 shuffle25[4];
        uint32_t thread_mix26;
        thread_mix26 = fnv_reduce(mix20[p]);
        shuffle25[0].x = __shfl_sync(4294967295U, (thread_mix26), (0), ((128 / 16)));
        shuffle25[0].y = __shfl_sync(4294967295U, (thread_mix26), (1), ((128 / 16)));
        shuffle25[1].x = __shfl_sync(4294967295U, (thread_mix26), (2), ((128 / 16)));
        shuffle25[1].y = __shfl_sync(4294967295U, (thread_mix26), (3), ((128 / 16)));
        shuffle25[2].x = __shfl_sync(4294967295U, (thread_mix26), (4), ((128 / 16)));
        shuffle25[2].y = __shfl_sync(4294967295U, (thread_mix26), (5), ((128 / 16)));
        shuffle25[3].x = __shfl_sync(4294967295U, (thread_mix26), (6), ((128 / 16)));
        shuffle25[3].y = __shfl_sync(4294967295U, (thread_mix26), (7), ((128 / 16)));
        if ((i + p) == thread_id17) {
            state16[8] = shuffle25[0];
            state16[9] = shuffle25[1];
            state16[10] = shuffle25[2];
            state16[11] = shuffle25[3];
        }
    }
}
if (!(cuda_swab64(keccak_f1600_final(state16)) > d_target)) {
    mix_hash14[0] = state16[8];
    mix_hash14[1] = state16[9];
    mix_hash14[2] = state16[10];
    mix_hash14[3] = state16[11];
    goto exit_point1;
}
uint32_t index19;
index19 = atomicInc((uint32_t *)&g_output9->count, 4294967295U);
if (index19 >= 4U)
    goto exit_point1;
g_output9->result[index19].gid = gid11;
g_output9->result[index19].mix[0] = mix12[0].x;
g_output9->result[index19].mix[1] = mix12[0].y;
g_output9->result[index19].mix[2] = mix12[1].x;
g_output9->result[index19].mix[3] = mix12[1].y;
g_output9->result[index19].mix[4] = mix12[2].x;
g_output9->result[index19].mix[5] = mix12[2].y;
g_output9->result[index19].mix[6] = mix12[3].x;
g_output9->result[index19].mix[7] = mix12[3].y;
exit_point1:
;
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
for (int i = 0; i < iter4; i++) {
    uint32_t nonce5;
    nonce5 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter4 + i + startNonce1;
    uint64_t m6[16];
    m6[0] = d_blake_data[0];
    m6[1] = d_blake_data[1];
    m6[2] = d_blake_data[2];
    m6[3] = d_blake_data[3];
    m6[4] = d_blake_data[4];
    m6[5] = d_blake_data[5];
    m6[6] = d_blake_data[6];
    m6[7] = d_blake_data[7];
    m6[8] = d_blake_data[8];
    ((uint32_t *)m6)[18] = *((uint32_t *)(&d_blake_data[9]));
    ((uint32_t *)m6)[19] = nonce5;
    m6[10] = m6[11] = 0;
    m6[12] = m6[13] = 0;
    m6[14] = m6[15] = 0;
    uint64_t v7[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
    G_b(0, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(0, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(0, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(0, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(0, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(0, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(0, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(0, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(1, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(1, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(1, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(1, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(1, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(1, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(1, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(1, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(2, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(2, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(2, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(2, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(2, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(2, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(2, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(2, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(3, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(3, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(3, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(3, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(3, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(3, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(3, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(3, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(4, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(4, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(4, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(4, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(4, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(4, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(4, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(4, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(5, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(5, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(5, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(5, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(5, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(5, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(5, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(5, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(6, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(6, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(6, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(6, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(6, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(6, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(6, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(6, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(7, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(7, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(7, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(7, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(7, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(7, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(7, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(7, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(8, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(8, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(8, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(8, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(8, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(8, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(8, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(8, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(9, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(9, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(9, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(9, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(9, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(9, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(9, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(9, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(10, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(10, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(10, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(10, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(10, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(10, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(10, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(10, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(11, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(11, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(11, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(11, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(11, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(11, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(11, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(11, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    uint2 last8;
    last8 = vectorize(v7[3] ^ v7[11] ^ 11912009170470909681UL);
    if (last8.y <= target23.y && last8.x <= target23.x) {
        resNonce2[1] = resNonce2[0];
        resNonce2[0] = nonce5;
    }
}
label_5:;
}
 __global__ __launch_bounds__(256, 3) void ethash_search_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_0(volatile Search_results *g_output9, uint64_t start_nonce10, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, uint32_t iter4)
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
uint32_t gid11;
gid11 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
uint2 mix12[4];
uint64_t nonce13;
nonce13 = start_nonce10 + gid11;
uint2 *mix_hash14;
mix_hash14 = mix12;
bool result15;
result15 = false;
uint2 state16[12];
state16[4] = vectorize(nonce13);
keccak_f1600_init(state16);
int thread_id17;
thread_id17 = threadIdx_x_1 & ((128 / 16) - 1);
int mix_idx18;
mix_idx18 = thread_id17 & 3;
for (int i = 0; i < (128 / 16); i += 4) {
    uint4 mix20[4];
    uint32_t offset21[4];
    uint32_t init022[4];
    for (int p = 0; p < 4; p++) {
        uint2 shuffle23[8];
        for (int j = 0; j < 8; j++) {
            shuffle23[j].x = __shfl_sync(4294967295U, (state16[j].x), (i + p), ((128 / 16)));
            shuffle23[j].y = __shfl_sync(4294967295U, (state16[j].y), (i + p), ((128 / 16)));
        }
        switch (mix_idx18) {
          case 0:
            mix20[p] = vectorize2(shuffle23[0], shuffle23[1]);
            break;
          case 1:
            mix20[p] = vectorize2(shuffle23[2], shuffle23[3]);
            break;
          case 2:
            mix20[p] = vectorize2(shuffle23[4], shuffle23[5]);
            break;
          case 3:
            mix20[p] = vectorize2(shuffle23[6], shuffle23[7]);
            break;
        }
        init022[p] = __shfl_sync(4294967295U, (shuffle23[0].x), (0), ((128 / 16)));
    }
    for (uint32_t a = 0; a < 64; a += 4) {
        int t24;
        t24 = bfe(a, 2U, 3U);
        for (uint32_t b = 0; b < 4; b++) {
            for (int p = 0; p < 4; p++) {
                offset21[p] = ((init022[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix20[p])[b])) % d_dag_size;
                offset21[p] = __shfl_sync(4294967295U, (offset21[p]), (t24), ((128 / 16)));
                mix20[p] = fnv4(mix20[p], d_dag[offset21[p]].uint4s[thread_id17]);
            }
        }
    }
    for (int p = 0; p < 4; p++) {
        uint2 shuffle25[4];
        uint32_t thread_mix26;
        thread_mix26 = fnv_reduce(mix20[p]);
        shuffle25[0].x = __shfl_sync(4294967295U, (thread_mix26), (0), ((128 / 16)));
        shuffle25[0].y = __shfl_sync(4294967295U, (thread_mix26), (1), ((128 / 16)));
        shuffle25[1].x = __shfl_sync(4294967295U, (thread_mix26), (2), ((128 / 16)));
        shuffle25[1].y = __shfl_sync(4294967295U, (thread_mix26), (3), ((128 / 16)));
        shuffle25[2].x = __shfl_sync(4294967295U, (thread_mix26), (4), ((128 / 16)));
        shuffle25[2].y = __shfl_sync(4294967295U, (thread_mix26), (5), ((128 / 16)));
        shuffle25[3].x = __shfl_sync(4294967295U, (thread_mix26), (6), ((128 / 16)));
        shuffle25[3].y = __shfl_sync(4294967295U, (thread_mix26), (7), ((128 / 16)));
        if ((i + p) == thread_id17) {
            state16[8] = shuffle25[0];
            state16[9] = shuffle25[1];
            state16[10] = shuffle25[2];
            state16[11] = shuffle25[3];
        }
    }
}
if (!(cuda_swab64(keccak_f1600_final(state16)) > d_target)) {
    mix_hash14[0] = state16[8];
    mix_hash14[1] = state16[9];
    mix_hash14[2] = state16[10];
    mix_hash14[3] = state16[11];
    goto exit_point1;
}
uint32_t index19;
index19 = atomicInc((uint32_t *)&g_output9->count, 4294967295U);
if (index19 >= 4U)
    goto exit_point1;
g_output9->result[index19].gid = gid11;
g_output9->result[index19].mix[0] = mix12[0].x;
g_output9->result[index19].mix[1] = mix12[0].y;
g_output9->result[index19].mix[2] = mix12[1].x;
g_output9->result[index19].mix[3] = mix12[1].y;
g_output9->result[index19].mix[4] = mix12[2].x;
g_output9->result[index19].mix[5] = mix12[2].y;
g_output9->result[index19].mix[6] = mix12[3].x;
g_output9->result[index19].mix[7] = mix12[3].y;
exit_point1:
;
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
for (int i = 0; i < iter4; i++) {
    uint32_t nonce5;
    nonce5 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * iter4 + i + startNonce1;
    uint64_t m6[16];
    m6[0] = d_blake_data[0];
    m6[1] = d_blake_data[1];
    m6[2] = d_blake_data[2];
    m6[3] = d_blake_data[3];
    m6[4] = d_blake_data[4];
    m6[5] = d_blake_data[5];
    m6[6] = d_blake_data[6];
    m6[7] = d_blake_data[7];
    m6[8] = d_blake_data[8];
    ((uint32_t *)m6)[18] = *((uint32_t *)(&d_blake_data[9]));
    ((uint32_t *)m6)[19] = nonce5;
    m6[10] = m6[11] = 0;
    m6[12] = m6[13] = 0;
    m6[14] = m6[15] = 0;
    uint64_t v7[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
    G_b(0, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(0, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(0, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(0, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(0, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(0, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(0, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(0, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(1, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(1, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(1, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(1, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(1, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(1, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(1, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(1, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(2, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(2, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(2, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(2, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(2, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(2, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(2, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(2, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(3, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(3, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(3, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(3, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(3, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(3, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(3, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(3, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(4, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(4, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(4, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(4, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(4, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(4, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(4, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(4, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(5, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(5, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(5, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(5, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(5, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(5, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(5, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(5, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(6, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(6, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(6, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(6, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(6, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(6, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(6, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(6, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(7, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(7, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(7, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(7, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(7, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(7, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(7, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(7, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(8, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(8, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(8, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(8, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(8, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(8, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(8, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(8, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(9, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(9, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(9, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(9, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(9, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(9, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(9, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(9, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(10, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(10, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(10, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(10, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(10, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(10, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(10, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(10, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    G_b(11, 0, v7[0], v7[4], v7[8], v7[12], m6);
    G_b(11, 1, v7[1], v7[5], v7[9], v7[13], m6);
    G_b(11, 2, v7[2], v7[6], v7[10], v7[14], m6);
    G_b(11, 3, v7[3], v7[7], v7[11], v7[15], m6);
    G_b(11, 4, v7[0], v7[5], v7[10], v7[15], m6);
    G_b(11, 5, v7[1], v7[6], v7[11], v7[12], m6);
    G_b(11, 6, v7[2], v7[7], v7[8], v7[13], m6);
    G_b(11, 7, v7[3], v7[4], v7[9], v7[14], m6);
    ;
    uint2 last8;
    last8 = vectorize(v7[3] ^ v7[11] ^ 11912009170470909681UL);
    if (last8.y <= target23.y && last8.x <= target23.x) {
        resNonce2[1] = resNonce2[0];
        resNonce2[0] = nonce5;
    }
}
label_7:;
}
