#include "include/matrix.cuh"
#include "assert.h"
#include <iostream>
#include <iomanip>
#include <chrono>

// GEMM 06 -- COARSE 2D VECTORIZED MAT_MUL IMPLEMENTATION
// SGEMM is C = α*(A @ B)+β*C; here α=1, β=0

#define COARSE_X 8
#define COARSE_Y 8
#define tiles_Arows 128
#define tiles_Acols 16
#define tiles_Bcols 128

__global__ void coarse2Dvec_mat_mul_kernel(float *d_A, float *d_B, float *d_C, int C_n_rows, int C_n_cols, int A_n_cols)
{
    // Number of threads per block
    const int num_threads = tiles_Arows * tiles_Bcols / (COARSE_X*COARSE_Y);
    static_assert(num_threads % tiles_Acols == 0);
    static_assert(num_threads % tiles_Bcols == 0);
    static_assert(tiles_Acols % 4 == 0);
    static_assert(tiles_Bcols % 4 == 0);
    assert(C_n_rows % 4 == 0);
    assert(C_n_cols % 4 == 0);
    assert(A_n_cols % 4 == 0);

    // Details regarding this thread
    const int b_y = blockIdx.y;
    const int b_x = blockIdx.x; 
    const int t_x = threadIdx.x;

    // 1D -> 2D while loading A
    const int A_view_tx = t_x % (tiles_Acols / 4);
    const int B_view_ty = t_x / (tiles_Bcols / 4);
    const int B_view_tx = t_x % (tiles_Bcols / 4);
    const int A_view_ty = t_x / (tiles_Acols / 4);

    // loading A and B
    const int stride_A = num_threads/(tiles_Acols / 4);
    const int stride_B = num_threads/(tiles_Bcols / 4);

    // Working on C[row, col]
    const int row = COARSE_Y * (t_x / (tiles_Bcols/COARSE_X));
    const int col = COARSE_X * (t_x % (tiles_Bcols/COARSE_X));
    const int num_tiles = ceil((float)A_n_cols/tiles_Acols);
    
    // Allocating shared memory
    __shared__ float sh_A[tiles_Acols][tiles_Arows];
    __shared__ float sh_B[tiles_Acols][tiles_Bcols];

    // Parallel mat mul
    float value[COARSE_Y][COARSE_X] = {0.0f};
    float register_A[COARSE_X] = {0.0f};
    float register_B[COARSE_Y] = {0.0f};

    for (int tile = 0; tile < num_tiles; tile++)
    {
        // Load Tiles into shared memory
        for (int load_offset = 0; load_offset < tiles_Arows; load_offset+=stride_A)
        {
            if ((b_y*tiles_Arows + load_offset+A_view_ty < C_n_rows) && (((tile*tiles_Acols+A_view_tx*4)) < A_n_cols))
            {
                float4 temp_A = reinterpret_cast<float4 *>(&d_A[(b_y*tiles_Arows + load_offset+A_view_ty)*A_n_cols + ((tile*tiles_Acols+A_view_tx*4))])[0];
                sh_A[A_view_tx*4+0][load_offset+A_view_ty] = temp_A.x;
                sh_A[A_view_tx*4+1][load_offset+A_view_ty] = temp_A.y;
                sh_A[A_view_tx*4+2][load_offset+A_view_ty] = temp_A.z;
                sh_A[A_view_tx*4+3][load_offset+A_view_ty] = temp_A.w;
            }
            else
            {
                sh_A[A_view_tx*4+0][load_offset+A_view_ty] = 0.0f;
                sh_A[A_view_tx*4+1][load_offset+A_view_ty] = 0.0f;
                sh_A[A_view_tx*4+2][load_offset+A_view_ty] = 0.0f;
                sh_A[A_view_tx*4+3][load_offset+A_view_ty] = 0.0f;
            }
            
        }
        
        for (int load_offset = 0; load_offset < tiles_Acols; load_offset+=stride_B)
        {
            if (((tile*tiles_Acols + B_view_ty+load_offset) < A_n_cols) && (((b_x*tiles_Bcols + B_view_tx*4)) < C_n_cols))
            {
                float4 temp_B = reinterpret_cast<float4 *>(&d_B[(tile*tiles_Acols + B_view_ty+load_offset)*C_n_cols + ((b_x*tiles_Bcols + B_view_tx*4))])[0];
                sh_B[B_view_ty+load_offset][B_view_tx*4+0] = temp_B.x;
                sh_B[B_view_ty+load_offset][B_view_tx*4+1] = temp_B.y;
                sh_B[B_view_ty+load_offset][B_view_tx*4+2] = temp_B.z;
                sh_B[B_view_ty+load_offset][B_view_tx*4+3] = temp_B.w;
            }
            else
            {
                sh_B[B_view_ty+load_offset][B_view_tx*4+0] = 0.0f;
                sh_B[B_view_ty+load_offset][B_view_tx*4+1] = 0.0f;
                sh_B[B_view_ty+load_offset][B_view_tx*4+2] = 0.0f;
                sh_B[B_view_ty+load_offset][B_view_tx*4+3] = 0.0f;
            }
            
        }
        __syncthreads();

        // calculate per-thread results
        for (int k = 0; k < tiles_Acols; ++k) 
        {
            // block into registers
            for (int i = 0; i < COARSE_Y; ++i)
                register_A[i] = sh_A[k][row+i];
            
            for (int i = 0; i < COARSE_X; ++i)
                register_B[i] = sh_B[k][col+i];
            
            for (int cy = 0; cy < COARSE_Y; ++cy) 
            {
                for (int cx = 0; cx < COARSE_X; ++cx) 
                    value[cy][cx] += register_A[cy] * register_B[cx];
            }
        }
        __syncthreads();
    }

    // Assigning calculated value
    for (int cy = 0; cy < COARSE_Y; ++cy)
    {
        for (int cx = 0; cx < COARSE_X; cx++)
        {
            if ((b_y*tiles_Arows+row+cy < C_n_rows) && (b_x*tiles_Bcols+col+cx < C_n_cols))
                d_C[(b_y*tiles_Arows+row+cy)*C_n_cols + (b_x*tiles_Bcols+col+cx)] = 1*value[cy][cx] + 0*d_C[(b_y*tiles_Arows+row+cy)*C_n_cols + (b_x*tiles_Bcols+col+cx)];
        }
    } 
}

void gemm6(float *d_A, float *d_B, float *d_C, int C_n_rows, int C_n_cols, int A_n_cols)
{
    // Kernel execution
    dim3 dim_grid(ceil(C_n_cols/(float)(tiles_Bcols)), ceil(C_n_rows/(float)(tiles_Arows)));
    dim3 dim_block(tiles_Arows*tiles_Bcols/(COARSE_X*COARSE_Y));
    coarse2Dvec_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, C_n_rows, C_n_cols, A_n_cols);
}

int main() {
    // Matrix sizes to be processed
    int sizes[] = {128, 256, 512, 1024};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    double times[numSizes];
    double gflops[numSizes];

    std::cout << "Verification Results:\n";
    for (int index = 0; index < numSizes; index++) {
        int size = sizes[index];

        float *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, size * size * sizeof(float));
        cudaMalloc((void **)&d_B, size * size * sizeof(float));
        cudaMalloc((void **)&d_C, size * size * sizeof(float));

        float *h_A = new float[size * size];
        float *h_B = new float[size * size];
        float *h_C_device = new float[size * size];
        float *h_C_host = new float[size * size];

        srand(42);
        for (int j = 0; j < size * size; j++) {
            h_A[j] = static_cast<float>(rand()) / RAND_MAX;
            h_B[j] = static_cast<float>(rand()) / RAND_MAX;
        }

        cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice);

        auto start = std::chrono::high_resolution_clock::now();
        gemm6(d_A, d_B, d_C, size, size, size);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        times[index] = elapsed.count();

        double ops = 2.0 * size * size * size;
        gflops[index] = ops / (1e9 * times[index]);

        cudaMemcpy(h_C_device, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += h_A[i * size + k] * h_B[k * size + j];
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

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        delete[] h_A;
        delete[] h_B;
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



    return 0;
}