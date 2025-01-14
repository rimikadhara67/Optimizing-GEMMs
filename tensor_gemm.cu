#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "utils.cuh"

using namespace nvcuda;

// Make sure constants are multiples of 16 for tensor cores
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Tensor core kernel for computing matrix multiplication
__global__ void naive_tensor_gemm_kernel(const half* __restrict__ d_A,
                                   const half* __restrict__ d_B,
                                   float* __restrict__ d_C,
                                   const int M, const int N, const int K) {
    // Using 2D grid for Tiling
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize the output to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, &d_A[aRow * K + aCol], K);
            wmma::load_matrix_sync(b_frag, &d_B[bRow * N + bCol], N);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store the output
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(&d_C[cRow * N + cCol], acc_frag, N, wmma::mem_row_major);
    }
}

void tensor_gemm(half* d_A, half* d_B, float* d_C, int M, int N, int K) {
    // Calculate grid and block dimensions
    int block_size = 128;
    dim3 block(block_size);
    dim3 grid((M + (WMMA_M * block_size / 32) - 1) / (WMMA_M * block_size / 32),
              (N + WMMA_N - 1) / WMMA_N);

    naive_tensor_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

int main() {
    // Make sure we have a device that supports CUDA
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    // Matrix sizes to be processed
    int sizes[] = {128, 256, 512, 1024};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    double times[numSizes];
    double gflops[numSizes];
    double bandwidth[numSizes];

    std::cout << "Running on GPU: " << props.name << "\n";
    std::cout << "Verification Results:\n";

    for (int index = 0; index < numSizes; index++) {
        int size = sizes[index];

        // Ensure dimensions are multiples of 16
        if (size % 16 != 0) {
            std::cout << "Size " << size << " is not a multiple of 16\n";
            continue;
        }

        // Allocate device memory
        half *d_A_ptr, *d_B_ptr;
        float *d_C_ptr;
        cudaMalloc(&d_A_ptr, size * size * sizeof(half));
        cudaMalloc(&d_B_ptr, size * size * sizeof(half));
        cudaMalloc(&d_C_ptr, size * size * sizeof(float));

        // Allocate host memory
        half *h_A_ptr = new half[size * size];
        half *h_B_ptr = new half[size * size];
        float *h_C_device = new float[size * size];
        float *h_C_host = new float[size * size];

        // Initialize matrices with random values
        for (int j = 0; j < size * size; j++) {
            h_A_ptr[j] = __float2half(static_cast<float>(rand()) / RAND_MAX);
            h_B_ptr[j] = __float2half(static_cast<float>(rand()) / RAND_MAX);
        }

        // Copy data to device
        cudaMemcpy(d_A_ptr, h_A_ptr, size * size * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_ptr, h_B_ptr, size * size * sizeof(half), cudaMemcpyHostToDevice);

        // Warm up
        tensor_gemm(d_A_ptr, d_B_ptr, d_C_ptr, size, size, size);
        cudaDeviceSynchronize();

        // Time the computation
        auto start = std::chrono::high_resolution_clock::now();
        tensor_gemm(d_A_ptr, d_B_ptr, d_C_ptr, size, size, size);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        times[index] = elapsed.count();

        // Calculate performance metrics
        double ops = 2.0 * size * size * size;
        gflops[index] = ops / (1e9 * times[index]);

        // Copy result back to host
        cudaMemcpy(h_C_device, d_C_ptr, size * size * sizeof(float), cudaMemcpyDeviceToHost);

        // Calculate bandwidth
        double dataTransferred = 2.0 * size * size * sizeof(half) + size * size * sizeof(float);
        bandwidth[index] = dataTransferred / (times[index] * 1e9);  // GB/s

        // Compute reference result on CPU
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += __half2float(h_A_ptr[i * size + k]) * __half2float(h_B_ptr[k * size + j]);
                }
                h_C_host[i * size + j] = sum;
            }
        }

        // Verify results
        bool correct = true;
        float maxError = 0.0f;
        for (int i = 0; i < size * size; i++) {
            float diff = fabs(h_C_host[i] - h_C_device[i]);
            maxError = max(maxError, diff);
            if (diff > 1e-1) {  // Using larger tolerance for FP16
                correct = false;
                break;
            }
        }

        std::cout << size << "x" << size << " matrix: ";
        if (correct) {
            std::cout << "\033[32mVerification Successful\033[0m (Max Error: " << maxError << ")\n";
        } else {
            std::cout << "\033[31mVerification Failed\033[0m (Max Error: " << maxError << ")\n";
        }

        // Cleanup
        cudaFree(d_A_ptr);
        cudaFree(d_B_ptr);
        cudaFree(d_C_ptr);
        delete[] h_A_ptr;
        delete[] h_B_ptr;
        delete[] h_C_device;
        delete[] h_C_host;
    }

    // Print performance results
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "Matrix Sizes:   ";
    for (int size : sizes) {
        std::cout << std::setw(15) << size;
    }
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "Time (Seconds): ";
    for (double time : times) {
        std::cout << std::setw(15) << std::fixed << std::setprecision(9) << time;
    }
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "GFLOPS:         ";
    for (double gflop : gflops) {
        std::cout << std::setw(15) << std::fixed << std::setprecision(6) << gflop;
    }
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "Bandwidth (GB/s):";
    for (double bw : bandwidth) {
        std::cout << std::setw(14) << std::fixed << std::setprecision(2) << bw << " ";
    }
    std::cout << "\n------------------------------------------------------------------------------\n";

    return 0;
}
