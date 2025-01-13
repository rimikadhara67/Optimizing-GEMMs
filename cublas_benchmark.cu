#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <chrono>

// Helper function to check cuBLAS errors
#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s %d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    // Matrix sizes to be processed
    int sizes[] = {128, 256, 512, 1024};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    double times[numSizes];
    double gflops[numSizes];
    double bandwidth[numSizes];

    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

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

        // Initialize matrices with random values
        srand(42);
        for (int j = 0; j < size * size; j++) {
            h_A_ptr[j] = static_cast<float>(rand()) / RAND_MAX;
            h_B_ptr[j] = static_cast<float>(rand()) / RAND_MAX;
        }

        cudaMemcpy(d_A_ptr, h_A_ptr, size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_ptr, h_B_ptr, size * size * sizeof(float), cudaMemcpyHostToDevice);

        // Alpha and beta values for GEMM
        float alpha = 1.0f;
        float beta = 0.0f;

        // Warmup run
        CHECK_CUBLAS(cublasSgemm(handle, 
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                size, size, size,
                                &alpha,
                                d_B_ptr, size,
                                d_A_ptr, size,
                                &beta,
                                d_C_ptr, size));
        cudaDeviceSynchronize();

        // Multiple timing runs
        const int NUM_RUNS = 10;
        float elapsed_time;
        
        cudaEventRecord(start_event);
        for (int run = 0; run < NUM_RUNS; run++) {
            CHECK_CUBLAS(cublasSgemm(handle, 
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    size, size, size,
                                    &alpha,
                                    d_B_ptr, size,
                                    d_A_ptr, size,
                                    &beta,
                                    d_C_ptr, size));
        }
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        
        // Convert to seconds and average over runs
        times[index] = (elapsed_time / 1000.0) / NUM_RUNS;  // Convert ms to s and average

        // Calculate performance metrics
        double ops = 2.0 * size * size * size;
        gflops[index] = ops / (1e9 * times[index]);
        
        cudaMemcpy(h_C_device, d_C_ptr, size * size * sizeof(float), cudaMemcpyDeviceToHost);
        double dataTransferred = 3 * size * size * sizeof(float);
        bandwidth[index] = dataTransferred / (times[index] * 1e9);

        // CPU verification
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += h_A_ptr[i * size + k] * h_B_ptr[k * size + j];
                }
                h_C_host[i * size + j] = sum;
            }
        }

        // Verify results
        bool correct = true;
        for (int i = 0; i < size * size; i++) {
            if (fabs(h_C_host[i] - h_C_device[i]) > 1e-3) {
                correct = false;
                std::cout << "Verification failed at index " << i << ": Host=" << h_C_host[i] 
                         << ", Device=" << h_C_device[i] << std::endl;
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

    // Print results table
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

    // Cleanup
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}