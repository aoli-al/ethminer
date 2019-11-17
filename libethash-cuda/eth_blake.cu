//
// Created by Leo Li on 2019-11-14.
//
#include "ethash_cuda_miner_kernel.h"

#include "ethash_cuda_miner_kernel_globals.h"

#include "cuda_helper.h"

#include "fnv.cuh"

#include <string.h>
#include <stdint.h>

#include "keccak.cuh"

#include "dagger_shuffled.cuh"

//#include <sph/blake2b.h>

#define TPB 512
#define NBN 2

static uint32_t *d_resNonces[MAX_GPUS];

__device__ uint64_t d_data[10];

#define AS_U32(addr)   *((uint32_t*)(addr))

static __constant__ const int8_t blake2b_sigma[12][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
    { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  } ,
    { 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  } ,
    { 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  } ,
    { 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 } ,
    { 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  } ,
    { 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 } ,
    { 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 } ,
    { 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  } ,
    { 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  } ,
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
    { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  }
};

// host mem align
#define A 64
//
//extern "C" void blake2b_hash(void *output, const void *input)
//{
//    uint8_t _ALIGN(A) hash[32];
//    blake2b_ctx ctx;
//
//    blake2b_init(&ctx, 32, NULL, 0);
//    blake2b_update(&ctx, input, 80);
//    blake2b_final(&ctx, hash);
//
//    memcpy(output, hash, 32);
//}

// ----------------------------------------------------------------

__device__ __forceinline__
static void G(const int r, const int i, uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t const m[16])
{
    a = a + b + m[ blake2b_sigma[r][2*i] ];
    ((uint2*)&d)[0] = SWAPUINT2( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
    c = c + d;
    ((uint2*)&b)[0] = ROR24( ((uint2*)&b)[0] ^ ((uint2*)&c)[0] );
    a = a + b + m[ blake2b_sigma[r][2*i+1] ];
    ((uint2*)&d)[0] = ROR16( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
    c = c + d;
    ((uint2*)&b)[0] = ROR2( ((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

#define ROUND(r) \
	G(r, 0, v[0], v[4], v[ 8], v[12], m); \
	G(r, 1, v[1], v[5], v[ 9], v[13], m); \
	G(r, 2, v[2], v[6], v[10], v[14], m); \
	G(r, 3, v[3], v[7], v[11], v[15], m); \
	G(r, 4, v[0], v[5], v[10], v[15], m); \
	G(r, 5, v[1], v[6], v[11], v[12], m); \
	G(r, 6, v[2], v[7], v[ 8], v[13], m); \
	G(r, 7, v[3], v[4], v[ 9], v[14], m);

__global__
void blake2b_gpu_hash(
    const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2)
{
    for (int i = 0; i < 20; i++)
    {
        const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) + startNonce;

        uint64_t m[16];

        m[0] = d_data[0];
        m[1] = d_data[1];
        m[2] = d_data[2];
        m[3] = d_data[3];
        m[4] = d_data[4];
        m[5] = d_data[5];
        m[6] = d_data[6];
        m[7] = d_data[7];
        m[8] = d_data[8];
        ((uint32_t*)m)[18] = AS_U32(&d_data[9]);
        ((uint32_t*)m)[19] = nonce;

        m[10] = m[11] = 0;
        m[12] = m[13] = 0;
        m[14] = m[15] = 0;

        uint64_t v[16] = {0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
            0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b,
            0x5be0cd19137e2179, 0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
            0xa54ff53a5f1d36f1, 0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294,
            0x5be0cd19137e2179};

        ROUND(0);
        ROUND(1);
        ROUND(2);
        ROUND(3);
        ROUND(4);
        ROUND(5);
        ROUND(6);
        ROUND(7);
        ROUND(8);
        ROUND(9);
        ROUND(10);
        ROUND(11);

        uint2 last = vectorize(v[3] ^ v[11] ^ 0xa54ff53a5f1d36f1);
        if (last.y <= target2.y && last.x <= target2.x)
        {
            resNonce[1] = resNonce[0];
            resNonce[0] = nonce;
        }
    }
}

#define copy(dst, src, count)        \
    for (int i = 0; i != count; ++i) \
    {                                \
        (dst)[i] = (src)[i];         \
    }


__global__ void ethash_search2(volatile Search_results* g_output, uint64_t start_nonce)
{
    uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint2 mix[4];
    uint64_t nonce = start_nonce + gid;
    uint2* mix_hash = mix;
    bool result = false;

    uint2 state[12];

    state[4] = vectorize(nonce);

    keccak_f1600_init(state);

    // Threads work together in this phase in groups of 8.
    const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
    const int mix_idx = thread_id & 3;

    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
    {
        uint4 mix[_PARALLEL_HASH];
        uint32_t offset[_PARALLEL_HASH];
        uint32_t init0[_PARALLEL_HASH];

        // share init among threads
        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[8];
            for (int j = 0; j < 8; j++)
            {
                shuffle[j].x = SHFL(state[j].x, i + p, THREADS_PER_HASH);
                shuffle[j].y = SHFL(state[j].y, i + p, THREADS_PER_HASH);
            }
            switch (mix_idx)
            {
            case 0:
                mix[p] = vectorize2(shuffle[0], shuffle[1]);
                break;
            case 1:
                mix[p] = vectorize2(shuffle[2], shuffle[3]);
                break;
            case 2:
                mix[p] = vectorize2(shuffle[4], shuffle[5]);
                break;
            case 3:
                mix[p] = vectorize2(shuffle[6], shuffle[7]);
                break;
            }
            init0[p] = SHFL(shuffle[0].x, 0, THREADS_PER_HASH);
        }

        for (uint32_t a = 0; a < ACCESSES; a += 4)
        {
            int t = bfe(a, 2u, 3u);

            for (uint32_t b = 0; b < 4; b++)
            {
                for (int p = 0; p < _PARALLEL_HASH; p++)
                {
                    offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t*)&mix[p])[b]) % d_dag_size;
                    offset[p] = SHFL(offset[p], t, THREADS_PER_HASH);
                    mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
                }
            }
        }

        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[4];
            uint32_t thread_mix = fnv_reduce(mix[p]);

            // update mix across threads
            shuffle[0].x = SHFL(thread_mix, 0, THREADS_PER_HASH);
            shuffle[0].y = SHFL(thread_mix, 1, THREADS_PER_HASH);
            shuffle[1].x = SHFL(thread_mix, 2, THREADS_PER_HASH);
            shuffle[1].y = SHFL(thread_mix, 3, THREADS_PER_HASH);
            shuffle[2].x = SHFL(thread_mix, 4, THREADS_PER_HASH);
            shuffle[2].y = SHFL(thread_mix, 5, THREADS_PER_HASH);
            shuffle[3].x = SHFL(thread_mix, 6, THREADS_PER_HASH);
            shuffle[3].y = SHFL(thread_mix, 7, THREADS_PER_HASH);

            if ((i + p) == thread_id)
            {
                // move mix into state:
                state[8] = shuffle[0];
                state[9] = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }

    // keccak_256(keccak_512(header..nonce) .. mix);
    if (!(cuda_swab64(keccak_f1600_final(state)) > d_target)) {
        mix_hash[0] = state[8];
        mix_hash[1] = state[9];
        mix_hash[2] = state[10];
        mix_hash[3] = state[11];
        return;
    }

    uint32_t index = atomicInc((uint32_t*)&g_output->count, 0xffffffff);
    if (index >= MAX_SEARCH_RESULTS)
        return;
    g_output->result[index].gid = gid;
    g_output->result[index].mix[0] = mix[0].x;
    g_output->result[index].mix[1] = mix[0].y;
    g_output->result[index].mix[2] = mix[1].x;
    g_output->result[index].mix[3] = mix[1].y;
    g_output->result[index].mix[4] = mix[2].x;
    g_output->result[index].mix[5] = mix[2].y;
    g_output->result[index].mix[6] = mix[3].x;
    g_output->result[index].mix[7] = mix[3].y;
}

__global__ void ethash_search_blake_fused(
    volatile Search_results* g_output, uint64_t start_nonce,
    const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2)
{
    if (threadIdx.x < 128) {
        uint32_t const gid = blockIdx.x * 128 + threadIdx.x;
        uint2 mix[4];
        if (compute_hash(start_nonce + gid, mix))
            return;
        uint32_t index = atomicInc((uint32_t*)&g_output->count, 0xffffffff);
        if (index >= MAX_SEARCH_RESULTS)
            return;
        g_output->result[index].gid = gid;
        g_output->result[index].mix[0] = mix[0].x;
        g_output->result[index].mix[1] = mix[0].y;
        g_output->result[index].mix[2] = mix[1].x;
        g_output->result[index].mix[3] = mix[1].y;
        g_output->result[index].mix[4] = mix[2].x;
        g_output->result[index].mix[5] = mix[2].y;
        g_output->result[index].mix[6] = mix[3].x;
        g_output->result[index].mix[7] = mix[3].y;
    } else {
        const uint32_t nonce = (512 * blockIdx.x + threadIdx.x - 128) + startNonce;
        for (int i = 0; i < 20; i++) {

            uint64_t m[16];

            m[0] = d_data[0];
            m[1] = d_data[1];
            m[2] = d_data[2];
            m[3] = d_data[3];
            m[4] = d_data[4];
            m[5] = d_data[5];
            m[6] = d_data[6];
            m[7] = d_data[7];
            m[8] = d_data[8];
            ((uint32_t*)m)[18] = AS_U32(&d_data[9]);
            ((uint32_t*)m)[19] = nonce;

            m[10] = m[11] = 0;
            m[12] = m[13] = 0;
            m[14] = m[15] = 0;

            uint64_t v[16] = {
                0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
                0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294, 0x5be0cd19137e2179
            };

            ROUND( 0);
            ROUND( 1);
            ROUND( 2);
            ROUND( 3);
            ROUND( 4);
            ROUND( 5);
            ROUND( 6);
            ROUND( 7);
            ROUND( 8);
            ROUND( 9);
            ROUND(10);
            ROUND(11);

            uint2 last = vectorize(v[3] ^ v[11] ^ 0xa54ff53a5f1d36f1);
            if (last.y <= target2.y && last.x <= target2.x) {
                resNonce[1] = resNonce[0];
                resNonce[0] = nonce;
            }

        }
    }
}

extern __global__ void ethash_search(volatile Search_results* g_output, uint64_t start_nonce);



void run_ethash_search_blake(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
                       volatile Search_results* g_output, uint64_t start_nonce)
{
    {
        uint32_t threads = 4194304;
        dim3 grid((threads + TPB-1)/TPB);
        dim3 block(TPB);
        auto thr_id = 0;
        CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], NBN * sizeof(uint32_t)));
        if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
            return;
        const uint2 target2 = make_uint2(0, 1);
        cudaStream_t t1;
        cudaStream_t t2;
        cudaStreamCreate ( &t1);
        cudaStreamCreate ( &t2);
        blake2b_gpu_hash <<<grid, block, 8, t1>>> (threads, 0, d_resNonces[thr_id], target2);
        ethash_search2<<<gridSize, blockSize, 0, t2>>>(g_output, start_nonce);
        ethash_search_blake_fused<<<grid, 640, 8>>>
        (g_output, start_nonce, threads, 0, d_resNonces[thr_id], target2);
    }



    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaGetLastError());
}

