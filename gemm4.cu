#include "include/matrix.cuh"
#include "assert.h"
#include <iostream>
#include <iomanip>
#include <chrono>

// GEMM 04 -- COARSE 1D MAT_MUL IMPLEMENTATION
// SGEMM is C = α*(A @ B)+β*C; here α=1, β=0

#define COARSE_FACTOR 8
#define tiles_Arows 64
#define tiles_Acols 8
#define tiles_Bcols 64

__global__ void coarse1D_mat_mul_kernel(float *d_A, float *d_B, float *d_C, 
                                    int C_n_rows, int C_n_cols, int A_n_cols)
{
    const int b_x = blockIdx.x;
    const int b_y = blockIdx.y;
    const int t_x = threadIdx.x;

    // 1D -> 2D
    const int A_view_ty = t_x / tiles_Acols;
    const int A_view_tx = t_x % tiles_Acols;
    const int B_view_ty = t_x / tiles_Bcols;
    const int B_view_tx = t_x % tiles_Bcols;

    // Defining rows and cols for C[row, col] and tiles
    const int row = tiles_Arows * b_y + COARSE_FACTOR * (t_x/tiles_Bcols);
    const int col = tiles_Bcols * b_x + (t_x % tiles_Bcols);
    const int num_tiles = ceil((float)A_n_cols / tiles_Acols);

    // Saving in SMEM
    __shared__ float sh_A[tiles_Arows][tiles_Acols];
    __shared__ float sh_B[tiles_Acols][tiles_Bcols];

    float dot_prod[COARSE_FACTOR] = {0.0f};
    for (int tile = 0; tile < num_tiles; tile++){
        if ((b_y * tiles_Arows + A_view_ty < C_n_rows) && ((tile * tiles_Acols + A_view_tx) < A_n_cols)){
            sh_A[A_view_ty][A_view_tx] = d_A[(b_y*tiles_Arows + A_view_ty)*A_n_cols + (tile * tiles_Acols + A_view_tx)]; }
        else
            { sh_A[A_view_ty][A_view_tx] = 0.0f; }
        
        if (((tile * tiles_Acols + B_view_ty) < A_n_cols) && (b_x * tiles_Bcols + B_view_tx < C_n_cols)){
            sh_B[B_view_ty][B_view_tx] = d_B[(tile*tiles_Acols + B_view_ty) * C_n_cols + (b_x * tiles_Bcols + B_view_tx)];
        }
        else
            { sh_B[B_view_ty][B_view_tx] = 0.0f; }
        __syncthreads();

        for (int k = 0; k < tiles_Acols; k++)
        {
            float B_val_register = sh_B[k][B_view_tx];
            // Dot product
            for (int c = 0; c < COARSE_FACTOR; c++)
                dot_prod[c] += sh_A[B_view_ty*COARSE_FACTOR+c][k] * B_val_register;  
        }
        __syncthreads();
    }

    // Storing and assigning
    for (int c = 0; c < COARSE_FACTOR; ++c){
        if ((row+c < C_n_rows) && (col < C_n_cols)){
            d_C[(row+c)*C_n_cols + (col)] = 1*dot_prod[c] + 0*d_C[(row+c)*C_n_cols + (col)];
        }
    } 
}

// function to invoke above CUDA kernel
void gemm4(float *d_A, float *d_B, float *d_C, int C_n_rows, int C_n_cols, int A_n_cols){
    dim3 dim_grid(ceil(C_n_cols/(float)(tiles_Bcols)), ceil(C_n_rows/(float)(tiles_Arows)));
    dim3 dim_block(tiles_Arows * tiles_Bcols/COARSE_FACTOR);
    coarse1D_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, C_n_rows, C_n_cols, A_n_cols);
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
        gemm4(d_A_ptr, d_B_ptr, d_C_ptr, size, size, size);
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