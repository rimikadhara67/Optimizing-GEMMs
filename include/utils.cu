#include "include/utils.cuh"

void init_mat(MatrixFP32 mat, float val)
{
    // Getting Matrix Dimension
    int n_rows = mat.n_rows; 
    int n_cols = mat.n_cols;

    // Initializing val to each location
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            mat.ptr[i*n_cols + j] = val;
        }
    }
}

void random_init_mat(MatrixFP32 mat, int MAX_VAL, int MIN_VAL)
{
    // Getting Matrix Dimension
    int n_rows = mat.n_rows; 
    int n_cols = mat.n_cols;

    // Initializing val to each location
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            mat.ptr[i*n_cols + j] = (float)(rand() % (MAX_VAL - MIN_VAL + 1) + MIN_VAL);
        }
    }
}

void print_mat(MatrixFP32 mat, bool full)
{
    // Matrix must be on host
    assert(mat.on_device == false && "For printing matrix must be on host");

    // Getting Matrix Dimension
    int n_rows = mat.n_rows; 
    int n_cols = mat.n_cols;

    if (full == true)
    {
        // Print Full Matrix
        for (int i = 0; i < n_rows; i++)
        {
            std::cout << "| ";
            for (int j = 0; j < n_cols; j++)
            {
                std::cout << std::setw(5) << std::to_string(mat.ptr[i*n_cols + j]) << " ";
            }
            std::cout << " |" << "\n";
        }
    }
    else
    {
        // Print Partial Matrix
        for (int i = 0; i < 5; i++)
        {
            std::cout << "| ";
            for (int j = 0; j < 5; j++)
            {
                if (i == 3)
                {
                    std::cout << std::setw(5) << "  ...   " << " ";
                }
                else
                {
                    if (j != 3)
                        std::cout << std::setw(5) << std::to_string(mat.ptr[i*n_cols + j]) << " ";
                    else
                        std::cout << std::setw(5) << "..." << " ";
                }
            }
            std::cout << " |" << "\n";

        }
    }
}

void assert_mat(MatrixFP32 A_mat, MatrixFP32 B_mat, float eps)
{
    // Matrices must be on host
    assert(A_mat.on_device == false && "For asserting matrix must be on host");
    assert(B_mat.on_device == false && "For asserting matrix must be on host");

    // Getting A Matrix Dimension
    int A_n_rows = A_mat.n_rows; 
    int A_n_cols = A_mat.n_cols;

    // Getting B Matrix Dimension
    int B_n_rows = B_mat.n_rows; 
    int B_n_cols = B_mat.n_cols;

    // Asserting that matrices have same dimensions
    assert (A_n_rows == B_n_rows && "A rows must be equal to B rows");
    assert (A_n_cols == B_n_cols && "A cols must be equal to B cols");

    for (int i = 0; i < A_n_rows; i++)
    {
        for (int j = 0; j < A_n_cols; j++)
        {
            if (fabs(A_mat.ptr[i*A_n_cols + j] - B_mat.ptr[i*B_n_cols + j]) > eps)
            {
                std::cerr << "Assertion failed for " << "row number: " << i << ", col number: " << j << ".\n"
                        << "Absolute Difference: " << fabs(A_mat.ptr[i*A_n_cols + j] - B_mat.ptr[i*B_n_cols + j]) << "\n";
                assert(fabs(A_mat.ptr[i*A_n_cols + j] - B_mat.ptr[i*B_n_cols + j]) < eps && "Assertion failed!");
            }
        }
    }
}

void update_benckmark_txt(const std::string& filename, const double recorded_times[], 
                        const double recorded_gflops[], const int mat_sizes[], 
                        const int n_sizes)
{
    // Opening File
    std::ofstream file(filename);
    if (!file.is_open()) 
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "Matrix Sizes" << ": ";
    for (int i = 0; i < n_sizes; i++)
    {
        if (i != n_sizes-1)
            file << mat_sizes[i] << " " ;
        else
            file << mat_sizes[i] << "\n \n" ;
    }

    file << "Time (Seconds)" << ": ";
    for (int i = 0; i < n_sizes; i++)
    {
        if (i != n_sizes-1)
            file << recorded_times[i] << " " ;
        else
            file << recorded_times[i] << "\n \n" ;
    }

    file << "GPLOPS" << ": ";
    for (int i = 0; i < n_sizes; i++)
    {
        if (i != n_sizes-1)
            file << recorded_gflops[i] << " " ;
        else
            file << recorded_gflops[i] << "\n \n" ;
    }
    
}