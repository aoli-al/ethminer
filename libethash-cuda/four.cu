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


#define TPB_blake 128
#define NBN 2

namespace
{
__device__ uint64_t dd_target[1];
static uint32_t* d_sha256_resNonces[MAX_GPUS] = {0};
__constant__ static uint32_t __align__(8) c_target[2];
const __constant__ uint32_t __align__(8) c_H256[8] = {0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U,
   0xA54FF53AU, 0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U};

static const uint32_t cpu_K[64] = {0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B,
   0x59F111F1, 0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74,
   0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F,
   0x4A7484AA, 0x5CB0A9DC, 0x76F988DA, 0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3,
   0xD5A79147, 0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354,
   0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819,
   0xD6990624, 0xF40E3585, 0x106AA070, 0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3,
   0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA,
   0xA4506CEB, 0xBEF9A3F7, 0xC67178F2};
__constant__ static uint32_t __align__(8) c_K[64];
__constant__ static uint32_t __align__(8) c_midstate76[8];
__constant__ static uint32_t __align__(8) c_dataEnd80[4];

#define xor3b(a, b, c) (a ^ b ^ c)

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
   return xor3b(ROTR32(x, 2), ROTR32(x, 13), ROTR32(x, 22));
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
   return xor3b(ROTR32(x, 6), ROTR32(x, 11), ROTR32(x, 25));
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
   return xor3b(ROTR32(x, 7), ROTR32(x, 18), (x >> 3));
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
   return xor3b(ROTR32(x, 17), ROTR32(x, 19), (x >> 10));
}


__device__ static void sha2_step1(uint32_t a, uint32_t b, uint32_t c, uint32_t& d, uint32_t e,
   uint32_t f, uint32_t g, uint32_t& h, uint32_t in, const uint32_t Kshared)
{
   uint32_t t1, t2;
   uint32_t vxandx = xandx(e, f, g);
   uint32_t bsg21 = bsg2_1(e);
   uint32_t bsg20 = bsg2_0(a);
   uint32_t andorv = andor32(a, b, c);

   t1 = h + bsg21 + vxandx + Kshared + in;
   t2 = bsg20 + andorv;
   d = d + t1;
   h = t1 + t2;
}

__device__ static void sha2_step2(uint32_t a, uint32_t b, uint32_t c, uint32_t& d, uint32_t e,
   uint32_t f, uint32_t g, uint32_t& h, uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
   uint32_t t1, t2;

   int pcidx1 = (pc - 2) & 0xF;
   int pcidx2 = (pc - 7) & 0xF;
   int pcidx3 = (pc - 15) & 0xF;

   uint32_t inx0 = in[pc];
   uint32_t inx1 = in[pcidx1];
   uint32_t inx2 = in[pcidx2];
   uint32_t inx3 = in[pcidx3];

   uint32_t ssg21 = ssg2_1(inx1);
   uint32_t ssg20 = ssg2_0(inx3);
   uint32_t vxandx = xandx(e, f, g);
   uint32_t bsg21 = bsg2_1(e);
   uint32_t bsg20 = bsg2_0(a);
   uint32_t andorv = andor32(a, b, c);

   in[pc] = ssg21 + inx2 + ssg20 + inx0;

   t1 = h + bsg21 + vxandx + Kshared + in[pc];
   t2 = bsg20 + andorv;
   d = d + t1;
   h = t1 + t2;
}

__device__ static void sha256_round_body(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
{
   uint32_t a = state[0];
   uint32_t b = state[1];
   uint32_t c = state[2];
   uint32_t d = state[3];
   uint32_t e = state[4];
   uint32_t f = state[5];
   uint32_t g = state[6];
   uint32_t h = state[7];

   sha2_step1(a, b, c, d, e, f, g, h, in[0], Kshared[0]);
   sha2_step1(h, a, b, c, d, e, f, g, in[1], Kshared[1]);
   sha2_step1(g, h, a, b, c, d, e, f, in[2], Kshared[2]);
   sha2_step1(f, g, h, a, b, c, d, e, in[3], Kshared[3]);
   sha2_step1(e, f, g, h, a, b, c, d, in[4], Kshared[4]);
   sha2_step1(d, e, f, g, h, a, b, c, in[5], Kshared[5]);
   sha2_step1(c, d, e, f, g, h, a, b, in[6], Kshared[6]);
   sha2_step1(b, c, d, e, f, g, h, a, in[7], Kshared[7]);
   sha2_step1(a, b, c, d, e, f, g, h, in[8], Kshared[8]);
   sha2_step1(h, a, b, c, d, e, f, g, in[9], Kshared[9]);
   sha2_step1(g, h, a, b, c, d, e, f, in[10], Kshared[10]);
   sha2_step1(f, g, h, a, b, c, d, e, in[11], Kshared[11]);
   sha2_step1(e, f, g, h, a, b, c, d, in[12], Kshared[12]);
   sha2_step1(d, e, f, g, h, a, b, c, in[13], Kshared[13]);
   sha2_step1(c, d, e, f, g, h, a, b, in[14], Kshared[14]);
   sha2_step1(b, c, d, e, f, g, h, a, in[15], Kshared[15]);

#pragma unroll
   for (int i = 0; i < 3; i++)
   {
       sha2_step2(a, b, c, d, e, f, g, h, in, 0, Kshared[16 + 16 * i]);
       sha2_step2(h, a, b, c, d, e, f, g, in, 1, Kshared[17 + 16 * i]);
       sha2_step2(g, h, a, b, c, d, e, f, in, 2, Kshared[18 + 16 * i]);
       sha2_step2(f, g, h, a, b, c, d, e, in, 3, Kshared[19 + 16 * i]);
       sha2_step2(e, f, g, h, a, b, c, d, in, 4, Kshared[20 + 16 * i]);
       sha2_step2(d, e, f, g, h, a, b, c, in, 5, Kshared[21 + 16 * i]);
       sha2_step2(c, d, e, f, g, h, a, b, in, 6, Kshared[22 + 16 * i]);
       sha2_step2(b, c, d, e, f, g, h, a, in, 7, Kshared[23 + 16 * i]);
       sha2_step2(a, b, c, d, e, f, g, h, in, 8, Kshared[24 + 16 * i]);
       sha2_step2(h, a, b, c, d, e, f, g, in, 9, Kshared[25 + 16 * i]);
       sha2_step2(g, h, a, b, c, d, e, f, in, 10, Kshared[26 + 16 * i]);
       sha2_step2(f, g, h, a, b, c, d, e, in, 11, Kshared[27 + 16 * i]);
       sha2_step2(e, f, g, h, a, b, c, d, in, 12, Kshared[28 + 16 * i]);
       sha2_step2(d, e, f, g, h, a, b, c, in, 13, Kshared[29 + 16 * i]);
       sha2_step2(c, d, e, f, g, h, a, b, in, 14, Kshared[30 + 16 * i]);
       sha2_step2(b, c, d, e, f, g, h, a, in, 15, Kshared[31 + 16 * i]);
   }

   state[0] += a;
   state[1] += b;
   state[2] += c;
   state[3] += d;
   state[4] += e;
   state[5] += f;
   state[6] += g;
   state[7] += h;
}

__device__ static void sha256_round_last(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
{
   uint32_t a = state[0];
   uint32_t b = state[1];
   uint32_t c = state[2];
   uint32_t d = state[3];
   uint32_t e = state[4];
   uint32_t f = state[5];
   uint32_t g = state[6];
   uint32_t h = state[7];

   sha2_step1(a, b, c, d, e, f, g, h, in[0], Kshared[0]);
   sha2_step1(h, a, b, c, d, e, f, g, in[1], Kshared[1]);
   sha2_step1(g, h, a, b, c, d, e, f, in[2], Kshared[2]);
   sha2_step1(f, g, h, a, b, c, d, e, in[3], Kshared[3]);
   sha2_step1(e, f, g, h, a, b, c, d, in[4], Kshared[4]);
   sha2_step1(d, e, f, g, h, a, b, c, in[5], Kshared[5]);
   sha2_step1(c, d, e, f, g, h, a, b, in[6], Kshared[6]);
   sha2_step1(b, c, d, e, f, g, h, a, in[7], Kshared[7]);
   sha2_step1(a, b, c, d, e, f, g, h, in[8], Kshared[8]);
   sha2_step1(h, a, b, c, d, e, f, g, in[9], Kshared[9]);
   sha2_step1(g, h, a, b, c, d, e, f, in[10], Kshared[10]);
   sha2_step1(f, g, h, a, b, c, d, e, in[11], Kshared[11]);
   sha2_step1(e, f, g, h, a, b, c, d, in[12], Kshared[12]);
   sha2_step1(d, e, f, g, h, a, b, c, in[13], Kshared[13]);
   sha2_step1(c, d, e, f, g, h, a, b, in[14], Kshared[14]);
   sha2_step1(b, c, d, e, f, g, h, a, in[15], Kshared[15]);

#pragma unroll 2
   for (int i = 0; i < 2; i++)
   {
       sha2_step2(a, b, c, d, e, f, g, h, in, 0, Kshared[16 + 16 * i]);
       sha2_step2(h, a, b, c, d, e, f, g, in, 1, Kshared[17 + 16 * i]);
       sha2_step2(g, h, a, b, c, d, e, f, in, 2, Kshared[18 + 16 * i]);
       sha2_step2(f, g, h, a, b, c, d, e, in, 3, Kshared[19 + 16 * i]);
       sha2_step2(e, f, g, h, a, b, c, d, in, 4, Kshared[20 + 16 * i]);
       sha2_step2(d, e, f, g, h, a, b, c, in, 5, Kshared[21 + 16 * i]);
       sha2_step2(c, d, e, f, g, h, a, b, in, 6, Kshared[22 + 16 * i]);
       sha2_step2(b, c, d, e, f, g, h, a, in, 7, Kshared[23 + 16 * i]);
       sha2_step2(a, b, c, d, e, f, g, h, in, 8, Kshared[24 + 16 * i]);
       sha2_step2(h, a, b, c, d, e, f, g, in, 9, Kshared[25 + 16 * i]);
       sha2_step2(g, h, a, b, c, d, e, f, in, 10, Kshared[26 + 16 * i]);
       sha2_step2(f, g, h, a, b, c, d, e, in, 11, Kshared[27 + 16 * i]);
       sha2_step2(e, f, g, h, a, b, c, d, in, 12, Kshared[28 + 16 * i]);
       sha2_step2(d, e, f, g, h, a, b, c, in, 13, Kshared[29 + 16 * i]);
       sha2_step2(c, d, e, f, g, h, a, b, in, 14, Kshared[30 + 16 * i]);
       sha2_step2(b, c, d, e, f, g, h, a, in, 15, Kshared[31 + 16 * i]);
   }

   sha2_step2(a, b, c, d, e, f, g, h, in, 0, Kshared[16 + 16 * 2]);
   sha2_step2(h, a, b, c, d, e, f, g, in, 1, Kshared[17 + 16 * 2]);
   sha2_step2(g, h, a, b, c, d, e, f, in, 2, Kshared[18 + 16 * 2]);
   sha2_step2(f, g, h, a, b, c, d, e, in, 3, Kshared[19 + 16 * 2]);
   sha2_step2(e, f, g, h, a, b, c, d, in, 4, Kshared[20 + 16 * 2]);
   sha2_step2(d, e, f, g, h, a, b, c, in, 5, Kshared[21 + 16 * 2]);
   sha2_step2(c, d, e, f, g, h, a, b, in, 6, Kshared[22 + 16 * 2]);
   sha2_step2(b, c, d, e, f, g, h, a, in, 7, Kshared[23 + 16 * 2]);
   sha2_step2(a, b, c, d, e, f, g, h, in, 8, Kshared[24 + 16 * 2]);
   sha2_step2(h, a, b, c, d, e, f, g, in, 9, Kshared[25 + 16 * 2]);
   sha2_step2(g, h, a, b, c, d, e, f, in, 10, Kshared[26 + 16 * 2]);
   sha2_step2(f, g, h, a, b, c, d, e, in, 11, Kshared[27 + 16 * 2]);
   sha2_step2(e, f, g, h, a, b, c, d, in, 12, Kshared[28 + 16 * 2]);
   sha2_step2(d, e, f, g, h, a, b, c, in, 13, Kshared[29 + 16 * 2]);

   state[6] += g;
   state[7] += h;
}

__device__ __forceinline__ uint64_t cuda_swab32ll(uint64_t x)
{
   return MAKE_ULONGLONG(cuda_swab32(_LODWORD(x)), cuda_swab32(_HIDWORD(x)));
}


__global__
   /*__launch_bounds__(256,3)*/
   void
   sha256d_gpu_hash_shared(const uint32_t threads, const uint32_t startNonce, uint32_t* resNonces, uint32_t iter)
{
   for (int i = 0; i < iter; i++)
   {
       const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) * iter + i;

       __shared__ uint32_t s_K[64 * 4];
       // s_K[thread & 63] = c_K[thread & 63];
       if (threadIdx.x < 64U)
           s_K[threadIdx.x] = c_K[threadIdx.x];

       if (thread < threads)
       {
           const uint32_t nonce = startNonce + thread;

           uint32_t dat[16];
           AS_UINT2(dat) = AS_UINT2(c_dataEnd80);
           dat[2] = c_dataEnd80[2];
           dat[3] = nonce;
           dat[4] = 0x80000000;
           dat[15] = 0x280;
#pragma unroll 10
           for (int i = 5; i < 15; i++)
               dat[i] = 0;

           uint32_t buf[8];
#pragma unroll 4
           for (int i = 0; i < 8; i += 2)
               AS_UINT2(&buf[i]) = AS_UINT2(&c_midstate76[i]);
           // for (int i=0; i<8; i++) buf[i] = c_midstate76[i];

           sha256_round_body(dat, buf, s_K);

           // second sha256

#pragma unroll 8
           for (int i = 0; i < 8; i++)
               dat[i] = buf[i];
           dat[8] = 0x80000000;
#pragma unroll 6
           for (int i = 9; i < 15; i++)
               dat[i] = 0;
           dat[15] = 0x100;

#pragma unroll 8
           for (int i = 0; i < 8; i++)
               buf[i] = c_H256[i];

           sha256_round_last(dat, buf, s_K);

           // valid nonces
           uint64_t high = cuda_swab32ll(((uint64_t*)buf)[3]);
           if (high <= c_target[0])
           {
               // printf("%08x %08x - %016llx %016llx - %08x %08x\n", buf[7], buf[6], high, d_target[0], c_target[1], c_target[0]);
               resNonces[1] = atomicExch(resNonces, nonce);
               // d_target[0] = high;
           }
       }
   }
}

static uint32_t* d_resNonces_blake[MAX_GPUS];

__device__ uint64_t d_blake_data[10];

#define AS_U32(addr) *((uint32_t*)(addr))

static __constant__ const int8_t blake2b_sigma_blake[12][16] = {
   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
   {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
   {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
   {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
   {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
   {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
   {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
   {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
   {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
   {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
   {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

// host mem align
#define A 64
//
// extern "C" void blake2b_hash(void *output, const void *input)
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

__device__ __forceinline__ static void G_b(const int r, const int i, uint64_t& a, uint64_t& b,
   uint64_t& c, uint64_t& d, uint64_t const m[16])
{
   a = a + b + m[blake2b_sigma_blake[r][2 * i]];
   ((uint2*)&d)[0] = SWAPUINT2(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
   c = c + d;
   ((uint2*)&b)[0] = ROR24(((uint2*)&b)[0] ^ ((uint2*)&c)[0]);
   a = a + b + m[blake2b_sigma_blake[r][2 * i + 1]];
   ((uint2*)&d)[0] = ROR16(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
   c = c + d;
   ((uint2*)&b)[0] = ROR2(((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

#define ROUND(r) \
	G_b(r, 0, v[0], v[4], v[8], v[12], m); \
	G_b(r, 1, v[1], v[5], v[9], v[13], m); \
	G_b(r, 2, v[2], v[6], v[10], v[14], m); \
	G_b(r, 3, v[3], v[7], v[11], v[15], m); \
	G_b(r, 4, v[0], v[5], v[10], v[15], m); \
	G_b(r, 5, v[1], v[6], v[11], v[12], m); \
	G_b(r, 6, v[2], v[7], v[8], v[13], m); \
	G_b(r, 7, v[3], v[4], v[9], v[14], m);

__global__ void blake2b_gpu_hash(
   const uint32_t threads, const uint32_t startNonce, uint32_t* resNonce, const uint2 target2, uint32_t iter)
{
   for (int i = 0; i < iter; i++)
   {
       const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) * iter + i + startNonce;

       uint64_t m[16];

       m[0] = d_blake_data[0];
       m[1] = d_blake_data[1];
       m[2] = d_blake_data[2];
       m[3] = d_blake_data[3];
       m[4] = d_blake_data[4];
       m[5] = d_blake_data[5];
       m[6] = d_blake_data[6];
       m[7] = d_blake_data[7];
       m[8] = d_blake_data[8];
       ((uint32_t*)m)[18] = AS_U32(&d_blake_data[9]);
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

#define TPB 128
#define NBN 2

static uint32_t *d_resNonces[MAX_GPUS];

__device__ uint64_t d_data2[10];

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
//extern "C" void sia_blake2b_hash(void *output, const void *input)
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
//
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

// simplified for the last round
__device__ __forceinline__
static void H(const int r, const int i, uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t const m[16])
{
   a = a + b + m[ blake2b_sigma[r][2*i] ];
   ((uint2*)&d)[0] = SWAPUINT2( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
   c = c + d;
   ((uint2*)&b)[0] = ROR24( ((uint2*)&b)[0] ^ ((uint2*)&c)[0] );
   a = a + b + m[ blake2b_sigma[r][2*i+1] ];
   ((uint2*)&d)[0] = ROR16( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
   c = c + d;
}

// we only check v[0] and v[8]
#define ROUND_F(r) \
	G(r, 0, v[0], v[4], v[ 8], v[12], m); \
	G(r, 1, v[1], v[5], v[ 9], v[13], m); \
	G(r, 2, v[2], v[6], v[10], v[14], m); \
	G(r, 3, v[3], v[7], v[11], v[15], m); \
	G(r, 4, v[0], v[5], v[10], v[15], m); \
	G(r, 5, v[1], v[6], v[11], v[12], m); \
	H(r, 6, v[2], v[7], v[ 8], v[13], m);

__global__
//__launch_bounds__(128, 8) /* to force 64 regs */
void sia_blake2b_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2, uint32_t iter)
{

   for (int i = 0; i < iter; i++) {
       const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) * iter + i + startNonce;
       __shared__ uint64_t s_target;
       if (!threadIdx.x) s_target = devectorize(target2);
       uint64_t m[16];

       m[0] = d_data2[0];
       m[1] = d_data2[1];
       m[2] = d_data2[2];
       m[3] = d_data2[3];
       m[4] = d_data2[4] | nonce;
       m[5] = d_data2[5];
       m[6] = d_data2[6];
       m[7] = d_data2[7];
       m[8] = d_data2[8];
       m[9] = d_data2[9];

       m[10] = m[11] = 0;
       m[12] = m[13] = m[14] = m[15] = 0;

       uint64_t v[16] = {
           0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
           0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
           0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
           0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294, 0x5be0cd19137e2179
       };

       ROUND( 0 );
       ROUND( 1 );
       ROUND( 2 );
       ROUND( 3 );
       ROUND( 4 );
       ROUND( 5 );
       ROUND( 6 );
       ROUND( 7 );
       ROUND( 8 );
       ROUND( 9 );
       ROUND( 10 );
       ROUND_F( 11 );

       uint64_t h64 = cuda_swab64(0x6a09e667f2bdc928 ^ v[0] ^ v[8]);
       if (h64 <= s_target) {
           resNonce[1] = resNonce[0];
           resNonce[0] = nonce;
           s_target = h64;
       }
   }
   // if (!nonce) printf("%016lx ", s_target);
}

__global__ __launch_bounds__(128, 8) void ethash_search(volatile Search_results* g_output, uint64_t start_nonce)
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

#include "blake2b_gpu_hash_sha256d_gpu_hash_shared_.inc"
#include "blake2b_gpu_hash_sia_blake2b_gpu_hash_.inc"
#include "ethash_search_blake2b_gpu_hash_.inc"
#include "sha256d_gpu_hash_shared_ethash_search_.inc"
#include "sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_.inc"
#include "sia_blake2b_gpu_hash_ethash_search_.inc"


}

void four(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
                          volatile Search_results* g_output, uint64_t start_nonce, int input)
{

   {
       uint32_t resNonces[NBN] = { UINT32_MAX, UINT32_MAX };
       uint32_t result = UINT32_MAX;

       uint32_t threads = 1048576;
       dim3 grid((threads + TPB-1)/TPB);
       dim3 block(TPB);
       int thr_id = 0;

       cudaMalloc(&d_resNonces[thr_id], NBN * sizeof(uint32_t));
       /* Check error on Ctrl+C or kill to prevent segfaults on exit */

       const uint2 target2 = make_uint2(3, 5);

//        uint32_t threads = 1048576;
//        dim3 grid((threads + TPB_blake -1)/ TPB_blake);
//        dim3 block(TPB_blake);
       const uint32_t threads_sha256 = 1048576;
       const uint32_t threadsperblock = 128;
       dim3 grid_sha256(threads_sha256 /threadsperblock);
       dim3 block_sha256(threadsperblock);
//        auto thr_id = 0;
       CUDA_SAFE_CALL(cudaMalloc(&d_resNonces_blake[thr_id], NBN * sizeof(uint32_t)));
       CUDA_SAFE_CALL(cudaMalloc(&d_sha256_resNonces[0], 2*sizeof(uint32_t)));


//        const uint2 target2 = make_uint2(0, 1);
       cudaStream_t t1;
       cudaStream_t t2;
       cudaStreamCreate ( &t1);
       cudaStreamCreate ( &t2);
       cudaStream_t t3;
       cudaStream_t t4;
       cudaStreamCreate ( &t3);
       cudaStreamCreate ( &t4);
       for (int i = 50; i  < 140; i++) {
        ethash_search<<<gridSize, blockSize, 0, t1>>>(
            g_output, start_nonce
            );
        sia_blake2b_gpu_hash <<<grid, block, 8, t2>>> (
            threads, 0, d_resNonces[thr_id], target2, i
            );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sia_blake2b_gpu_hash_ethash_search_fused_kernel_vfuse_lb_idx_0
        <<<grid, 128, 8, t2>>> (
            threads, 0, d_resNonces[thr_id], target2, i,
            g_output, start_nonce
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sia_blake2b_gpu_hash_ethash_search_fused_kernel_vfuse_idx_0
        <<<grid, 128, 8, t2>>> (
            threads, 0, d_resNonces[thr_id], target2, i,
            g_output, start_nonce
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sia_blake2b_gpu_hash_ethash_search_fused_kernel_hfuse_idx_0
        <<<grid, 256, 8, t2>>> (
            threads, 0, d_resNonces[thr_id], target2, i,
            g_output, start_nonce
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sia_blake2b_gpu_hash_ethash_search_fused_kernel_hfuse_lb_idx_0
        <<<grid, 256, 8, t2>>> (
            threads, 0, d_resNonces[thr_id], target2, i,
            g_output, start_nonce
        );
       }
       for (int i = 50; i  < 140; i++) {
        ethash_search<<<gridSize, blockSize, 0, t1>>>(
            g_output, start_nonce
            );
        blake2b_gpu_hash <<<grid, block, 8, t3>>> (
            threads, 0, d_resNonces_blake[thr_id], target2, i
            );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        ethash_search_blake2b_gpu_hash_fused_kernel_vfuse_lb_idx_0
        <<<gridSize, blockSize, 8, t1>>> (
          g_output, start_nonce,
          threads, 0, d_resNonces_blake[thr_id], target2, i
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        ethash_search_blake2b_gpu_hash_fused_kernel_vfuse_idx_0
        <<<gridSize, blockSize, 8, t1>>> (
          g_output, start_nonce,
          threads, 0, d_resNonces_blake[thr_id], target2, i
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        ethash_search_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_0
        <<<gridSize, 256, 8, t1>>> (
          g_output, start_nonce,
          threads, 0, d_resNonces_blake[thr_id], target2, i
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        ethash_search_blake2b_gpu_hash_fused_kernel_hfuse_idx_0
        <<<gridSize, 256, 8, t1>>> (
          g_output, start_nonce,
          threads, 0, d_resNonces_blake[thr_id], target2, i
        );
       }
       for (int i = 50; i  < 140; i++) {
        ethash_search<<<gridSize, blockSize, 0, t1>>>(
            g_output, start_nonce
            );
        sha256d_gpu_hash_shared <<<grid_sha256, block_sha256, 0, t4>>> (
            threads_sha256 * i, 0, d_sha256_resNonces[0], i
            );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sha256d_gpu_hash_shared_ethash_search_fused_kernel_vfuse_lb_idx_0
        <<<gridSize, 128, 0, t1>>> (
            threads_sha256 * i, 0, d_sha256_resNonces[0], i,
            g_output, start_nonce
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sha256d_gpu_hash_shared_ethash_search_fused_kernel_vfuse_idx_0
        <<<gridSize, 128, 0, t1>>> (
            threads_sha256 * i, 0, d_sha256_resNonces[0], i,
            g_output, start_nonce
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_idx_0
        <<<gridSize, 256, 0, t1>>> (
            threads_sha256 * i, 0, d_sha256_resNonces[0], i,
            g_output, start_nonce
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sha256d_gpu_hash_shared_ethash_search_fused_kernel_hfuse_lb_idx_0
        <<<gridSize, 256, 0, t1>>> (
            threads_sha256 * i, 0, d_sha256_resNonces[0], i,
            g_output, start_nonce
        );
       }
       for (int i = 70; i  < 210; i++) {
        blake2b_gpu_hash <<<grid, block, 8, t3>>> (
            threads, 0, d_resNonces_blake[thr_id], target2, i
            );
       sia_blake2b_gpu_hash <<<grid, block, 8, t2>>> (
           threads, 0, d_resNonces[thr_id], target2, 140
           );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        blake2b_gpu_hash_sia_blake2b_gpu_hash_fused_kernel_vfuse_lb_idx_0
        <<<grid, 128, 8, t2>>> (
            threads, 0, d_resNonces_blake[thr_id], target2, i,
           threads, 0, d_resNonces[thr_id], target2, 140
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        blake2b_gpu_hash_sia_blake2b_gpu_hash_fused_kernel_vfuse_idx_0
        <<<grid, 128, 8, t2>>> (
            threads, 0, d_resNonces_blake[thr_id], target2, i,
           threads, 0, d_resNonces[thr_id], target2, 140
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        blake2b_gpu_hash_sia_blake2b_gpu_hash_fused_kernel_hfuse_idx_0
        <<<grid, 256, 8, t2>>> (
            threads, 0, d_resNonces_blake[thr_id], target2, i,
           threads, 0, d_resNonces[thr_id], target2, 140
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        blake2b_gpu_hash_sia_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_0
        <<<grid, 256, 8, t2>>> (
            threads, 0, d_resNonces_blake[thr_id], target2, i,
           threads, 0, d_resNonces[thr_id], target2, 140
        );
       }
       for (int i = 70; i  < 210; i++) {
        blake2b_gpu_hash <<<grid, block, 8, t3>>> (
            threads, 0, d_resNonces_blake[thr_id], target2, i
            );
       sha256d_gpu_hash_shared <<<grid_sha256, block_sha256, 0, t4>>> (
           threads_sha256 * 100, 0, d_sha256_resNonces[0], 100
           );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        blake2b_gpu_hash_sha256d_gpu_hash_shared_fused_kernel_vfuse_lb_idx_0
        <<<grid, 128, 8, t3>>> (
          threads, 0, d_resNonces_blake[thr_id], target2, i,
          threads_sha256 * 100, 0, d_sha256_resNonces[0], 100
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        blake2b_gpu_hash_sha256d_gpu_hash_shared_fused_kernel_vfuse_idx_0
        <<<grid, 128, 8, t3>>> (
          threads, 0, d_resNonces_blake[thr_id], target2, i,
          threads_sha256 * 100, 0, d_sha256_resNonces[0], 100
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        blake2b_gpu_hash_sha256d_gpu_hash_shared_fused_kernel_hfuse_idx_0
        <<<grid, 256, 8, t3>>> (
          threads, 0, d_resNonces_blake[thr_id], target2, i,
          threads_sha256 * 100, 0, d_sha256_resNonces[0], 100
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        blake2b_gpu_hash_sha256d_gpu_hash_shared_fused_kernel_hfuse_lb_idx_0
        <<<grid, 256, 8, t3>>> (
          threads, 0, d_resNonces_blake[thr_id], target2, i,
          threads_sha256 * 100, 0, d_sha256_resNonces[0], 100
        );
       }
       for (int i = 70; i  < 210; i++) {
        sia_blake2b_gpu_hash <<<grid, block, 8, t2>>> (
            threads, 0, d_resNonces[thr_id], target2, i
            );
        sha256d_gpu_hash_shared <<<grid_sha256, block_sha256, 0, t4>>> (
            threads_sha256 * 100, 0, d_sha256_resNonces[0], 100
            );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_vfuse_lb_idx_0
        <<<grid, 128, 8, t2>>> (
            threads_sha256 * 100, 0, d_sha256_resNonces[0], 100,
            threads, 0, d_resNonces[thr_id], target2, i
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_vfuse_idx_0
        <<<grid, 128, 8, t2>>> (
            threads_sha256 * 100, 0, d_sha256_resNonces[0], 100,
            threads, 0, d_resNonces[thr_id], target2, i
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_idx_0
        <<<grid, 256, 8, t2>>> (
            threads_sha256 * 100, 0, d_sha256_resNonces[0], 100,
            threads, 0, d_resNonces[thr_id], target2, i
        );
         CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
         if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
         if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
             return;
        cudaDeviceSynchronize();
        g_output->count = 0;
        sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_hfuse_lb_idx_0
        <<<grid, 256, 8, t2>>> (
            threads_sha256 * 100, 0, d_sha256_resNonces[0], 100,
            threads, 0, d_resNonces[thr_id], target2, i
        );
       }

       // ethash_search<<<gridSize, blockSize, 0, t1>>>(
           // g_output, start_nonce
           // );
       // int i = input;
       // sia_blake2b_gpu_hash <<<grid, block, 8, t2>>> (
           // threads, 0, d_resNonces[thr_id], target2, i
           // );
       // blake2b_gpu_hash <<<grid, block, 8, t3>>> (
           // threads, 0, d_resNonces_blake[thr_id], target2, i
           // );
      // const int iter_sha256 = input;
       // sha256d_gpu_hash_shared <<<grid_sha256, block_sha256, 0, t4>>> (
           // threads_sha256 * iter_sha256, 0, d_sha256_resNonces[0], iter_sha256
           // );

       cudaThreadSynchronize();
   }
   CUDA_SAFE_CALL(cudaGetLastError());
}
