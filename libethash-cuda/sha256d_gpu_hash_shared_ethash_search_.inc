 __global__ __launch_bounds__(128, 3) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_vfuse_lb_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(128, 0) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_vfuse_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_idx_1(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_idx_2(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_idx_3(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_idx_4(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_idx_5(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 0) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_idx_6(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 2) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_lb_idx_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 2) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_lb_idx_1(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 3) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_lb_idx_2(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 3) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_lb_idx_3(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 4) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_lb_idx_4(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 3) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_lb_idx_5(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
 __global__ __launch_bounds__(256, 3) void sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_lb_idx_6(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, uint32_t iter3, volatile Search_results *g_output10, uint64_t start_nonce11)
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
    const uint32_t gid12 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix13[4];
    uint64_t nonce14 = start_nonce11 + gid12;
    uint2 *mix_hash15 = mix13;
    bool result16 = false;
    uint2 state17[12];
    state17[4] = vectorize(nonce14);
    keccak_f1600_init(state17);
    const int thread_id18 = threadIdx_x_1 & ((128 / 16) - 1);
    const int mix_idx19 = thread_id18 & 3;
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
            int t25 = bfe(a, 2U, 3U);
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
            uint32_t thread_mix27 = fnv_reduce(mix21[p]);
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
