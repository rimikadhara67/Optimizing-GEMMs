#include "include/matrix.cuh"
#include "assert.h"
#include <iostream>
#include <iomanip>
#include <chrono>

// GEMM 01 -- NAIVE IMPLEMENTATION
// SGEMM is C = α*(A @ B)+β*C; here α=1, β=0
__global__ void naive_mat_mul_kernel(float *d_A, float *d_B, float *d_C, 
                                    int C_n_rows, int C_n_cols, int A_n_cols)
{
    const int row = blockDim.x * blockIdx.x + threadIdx.x;  // row from current thread
    const int col = blockDim.y * blockIdx.y + threadIdx.y;  // col from current thread

    // C[row][col]
    // Setting bounds
    if (row < C_n_rows && col < C_n_cols){
        float dot_prod = 0;
        // Computing dot product from A_row and B_col
        for (int k = 0; k < A_n_cols; k++){
            dot_prod += d_A[row * A_n_cols + k] * d_B[k * C_n_cols + col];
        }
        // Resulting C matrix = alpha(dot_prod) + beta[C_mtx]
        // alpha = 1 ; beta = 0 ;
        d_C[row * C_n_cols + col] = (1 * dot_prod) + 0 * d_C[row * C_n_cols + col];
    }
}

// function to invoke above CUDA kernel
void gemm1(float *d_A, float *d_B, float *d_C, int C_n_rows, int C_n_cols, int A_n_cols){
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(C_n_rows / (float)32), ceil(C_n_cols / (float)32), 1);
    naive_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, C_n_rows, C_n_cols, A_n_cols);
}

//  THIS MAIN BLOCK STAYS THE SAME FOR ALL GeMM IMPLEMENTATIONS
int main() {
    // Matrix sizes to be processed
    int sizes[] = {128, 256, 512, 1024};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    double times[numSizes];
    double gflops[numSizes];
    double bandwidth[numSizes];

    std::cout << "Verification Results:\n";
    for (int index = 0; index < numSizes; index++) {
        int size = sizes[index];

        float *d_A_ptr, *d_B_ptr, *d_C_ptr;
        cudaMalloc((void **)&d_A_ptr, size * size * sizeof(float));
        cudaMalloc((void **)&d_B_ptr, size * size * sizeof(float));
        cudaMalloc((void **)&d_C_ptr, size * size * sizeof(float));

        float *h_A_ptr = new float[size * size];
        float *h_B_ptr = new float[size * size];
        float *h_C_device = new float[size * size];
        float *h_C_host = new float[size * size];

        srand(42);
        for (int j = 0; j < size * size; j++) {
            h_A_ptr[j] = static_cast<float>(rand()) / RAND_MAX;
            h_B_ptr[j] = static_cast<float>(rand()) / RAND_MAX;
        }

        cudaMemcpy(d_A_ptr, h_A_ptr, size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_ptr, h_B_ptr, size * size * sizeof(float), cudaMemcpyHostToDevice);

        auto start = std::chrono::high_resolution_clock::now();
        gemm1(d_A_ptr, d_B_ptr, d_C_ptr, size, size, size);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        times[index] = elapsed.count();

        double ops = 2.0 * size * size * size;
        gflops[index] = ops / (1e9 * times[index]);

        cudaMemcpy(h_C_device, d_C_ptr, size * size * sizeof(float), cudaMemcpyDeviceToHost);
        double dataTransferred = 3 * size * size * sizeof(float);
        bandwidth[index] = dataTransferred / (times[index] * 1e9);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += h_A_ptr[i * size + k] * h_B_ptr[k * size + j];
                }
                h_C_host[i * size + j] = sum;
            }
        }

        bool correct = true;
        for (int i = 0; i < size * size; i++) {
            if (fabs(h_C_host[i] - h_C_device[i]) > 1e-4) {
                correct = false;
                std::cout << "Verification failed at index " << i << ": Host=" << h_C_host[i] << ", Device=" << h_C_device[i] << std::endl;
                break;
            }
        }

        std::cout << size << "x" << size << " matrix: ";
        if (correct) {
            std::cout << "\033[32mVerification Successful\033[0m\n";
        } else {
            std::cout << "\033[31mVerification Failed\033[0m\n";
        }

        cudaFree(d_A_ptr);
        cudaFree(d_B_ptr);
        cudaFree(d_C_ptr);

        delete[] h_A_ptr;
        delete[] h_B_ptr;
        delete[] h_C_device;
        delete[] h_C_host;
    }
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "Matrix Sizes:   ";
    for (int size : sizes) {
        std::cout << std::setw(15) << size;
    }
    std::cout << "\n";
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "Time (Seconds): ";
    for (double time : times) {
        std::cout << std::setw(15) << std::fixed << std::setprecision(9) << time;
    }
    std::cout << "\n";
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