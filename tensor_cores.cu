using namespace nvcuda;

// trying to use Tensor Cores in order to compute 16x16 matrix with half precision
__global__ void mat_mul_kernel(half *d_A, half d_B, float d_C, 
                                int C_n_rows, int C_n_cols, int A_n_cols)
    // usign 2D grid for Tiling
    int warpM = blockIdx.x;
    int warpN = blockIdx.y;
    // Declare 16x16 fragments
    //using wmma api
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; //memory allocation for tensor cores
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> b_frag; // same for b tile
    wmma::fragment<wmma::matrix_a, 16, 16, 16, float> c_frag; // same for result tile
    wmma::fill_fragment(c_frag, 0.0f); // init C tile with 0's

    // Loop over A_n_cols
    for (int i = 0; i < A_n_cols; i += 16){
        int aRow = warpM * 16;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * 16;

        // Load inputs
        wmma::load_matrix_sync(a_frag, &d_A[aRow * A_n_cols + aCol], A_n_cols);
        wmma::load_matrix_sync(b_frag, &d_B[bRow * C_n_cols + bCol], C_n_cols);

        // do MatMul in just a single line of code
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int cRow = warpM * 16;
    int cCol = warpN * 16;
    wmma::store_matrix_sync(&d_C[cRow * C_n_cols + cCol], c_frag, C_n_cols, wmma::mem_row_makor);
