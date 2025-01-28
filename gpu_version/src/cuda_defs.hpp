// cuda_defs.hpp
#ifndef CUDA_DEFS_HPP
#define CUDA_DEFS_HPP

#ifdef __CUDACC__
#define HOSTDEV __host__ __device__
#else
#define HOSTDEV
#endif

#endif // CUDA_DEFS_HPP
