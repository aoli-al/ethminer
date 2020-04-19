#define AS_U32(addr) *((uint32_t*)(addr))

__global__
void blake2b_gpu_hash(
    const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2, const uint32_t iter)
{
    for (int i = 0; i < iter; i++)
    {
        const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) * iter + i + startNonce;

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