#include "include/matrix.cuh"
#include "assert.h"
#include <iostream>
#include <iomanip>
#include <chrono>

// GEMM 03 -- TILED MAT_MUL IMPLEMENTATION
// SGEMM is C = α*(A @ B)+β*C; here α=1, β=0
#define TILE_WIDTH 32

__global__ void tiled_mat_mul_kernel(float *d_A, float *d_B, float *d_C, 
                                    int C_n_rows, int C_n_cols, int A_n_cols)
{
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
    const int b_x = blockIdx.x;
    const int b_y = blockIdx.y;
    const int t_x = threadIdx.x;
    const int t_y = threadIdx.y;

    // Initializing row, col and number of tiles
    const int row = TILE_WIDTH * b_y + t_y;
    const int col = TILE_WIDTH * b_x + t_x;
    const int num_tiles = ceil((float)A_n_cols/TILE_WIDTH);
    
    // Shared Memory allocation
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float dot_prod = 0;
    for (int tile = 0; tile < num_tiles; tile++){
        // loading our tiles onto shared memory
        // Matrix A
        if ((row < C_n_rows) && ((tile * TILE_WIDTH + t_x) < A_n_cols)){
            sh_A[t_y][t_x] = d_A[(row) * A_n_cols + (tile * TILE_WIDTH + t_x)]; }
        else { sh_A[t_y][t_x] = 0.0f; }

        // Matrix B
        if (((tile * TILE_WIDTH + t_y) < A_n_cols) && (col < C_n_cols)){
            sh_B[t_y][t_x] = d_B[(tile * TILE_WIDTH + t_y) * C_n_cols + (col)]; }
        else { sh_B[t_y][t_x] = 0.0f; }

        // sync threads
        __syncthreads();

        // calc dot product
        for (int k_tile = 0; k_tile < TILE_WIDTH; k_tile++)
            dot_prod += sh_A[t_y][k_tile] * sh_B[k_tile][t_x];
        __syncthreads();
    }

    // Storing and assigning
    if ((row < C_n_rows) && (col < C_n_cols))
        d_C[(row)*C_n_cols + (col)] =  1*dot_prod + 0*d_C[(row)*C_n_cols + (col)];
}

// function to invoke above CUDA kernel
void gemm3(float *d_A, float *d_B, float *d_C, int C_n_rows, int C_n_cols, int A_n_cols){
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(C_n_rows / (float)32), ceil(C_n_cols / (float)32), 1);
    tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, C_n_rows, C_n_cols, A_n_cols);
}

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
        gemm3(d_A_ptr, d_B_ptr, d_C_ptr, size, size, size);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        times[index] = elapsed.count();

        double ops = 2.0 * size * size * size;
        gflops[index] = ops / (1e9 * times[index]);

        cudaMemcpy(h_C_device, d_C_ptr, size * size * sizeof(float), cudaMemcpyDeviceToHost);
        double dataTransferred = 3 * size * size * sizeof(float);
        bandwidth[index] = (times[index] > 0) ? (dataTransferred / times[index] / 1e8) : 0; // computed in GB/s, handling zero duration

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