      #define CALL(iii, type, thread) cudaDeviceSynchronize();\
      CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));\
      if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      g_output->count = 0;\
      sia_blake2b_gpu_hash_ethash_search_fused_kernel_##type##_idx_##iii\
      <<<grid, thread, 8, t2>>> (\
          threads, 0, d_resNonces[thr_id], target2, i,\
          g_output, start_nonce\
      );\
      cudaDeviceSynchronize();\
      CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));\
      if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      g_output->count = 0;\
    ethash_search_blake2b_gpu_hash_fused_kernel_##type##_idx_##iii\
    <<<gridSize, thread, 8, t1>>> (\
        g_output, start_nonce,\
        threads, 0, d_resNonces_blake[thr_id], target2, i\
    );\
      cudaDeviceSynchronize();\
      CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));\
      if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      g_output->count = 0;\
    blake2b_gpu_hash_sia_blake2b_gpu_hash_fused_kernel_##type##_idx_##iii\
    <<<grid, thread, 8, t2>>> (\
        threads, 0, d_resNonces_blake[thr_id], target2, i,\
        threads, 0, d_resNonces[thr_id], target2, BLAKE_MID\
    );\
      cudaDeviceSynchronize();\
      CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));\
      if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      g_output->count = 0;\
    blake2b_gpu_hash_sha256d_gpu_hash_shared_fused_kernel_##type##_idx_##iii\
    <<<grid, thread, 8, t3>>> (\
        threads, 0, d_resNonces_blake[thr_id], target2, i,\
        threads_sha256 * SHA256_MID, 0, d_sha256_resNonces[0], SHA256_MID\
    );\
      cudaDeviceSynchronize();\
      CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));\
      if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      g_output->count = 0;\
    sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_fused_kernel_##type##_idx_##iii\
        <<<grid, thread, 8, t2>>> (\
            threads_sha256 * SHA256_MID, 0, d_sha256_resNonces[0], SHA256_MID,\
            threads, 0, d_resNonces[thr_id], target2, i\
        )\
    
#define CALLSH(iii, type, thread) cudaDeviceSynchronize();\
      CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));\
      if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)\
          return;\
      g_output->count = 0;\
        sha256d_gpu_hash_shared_ethash_search_fused_kernel_##type##_idx_##iii\
        <<<gridSize, thread, 0, t1>>> (\
            threads_sha256 * i, 0, d_sha256_resNonces[0], i,\
            g_output, start_nonce\
        )