# Optimizing GeMMs

A project on optimizing and accelerating an FP32 General Matrix Multiplications (GeMMs). With this project, I hope to start from a naive implementation of an FP32 GeMM and optimize it with methods such as coalescing, tiling, coarse and vectorization. I implement the code on 128, 256, 512, and 1024 threads. The results of those individually can be found in results/ folder.

List of all the optimizations:
* GeMM-1 Naive GeMM
* GeMM-2 Coalesced GeMM
* GeMM-3 Tiled GeMM
* GeMM-4 Coarse 1D GeMM
* GeMM-5 Coarse 2D GeMM
* GeMM-6 Coarse 2D Vectorized GeMM

## Results
### Comparing results for 1024 Threads
![Performance Optimization for 1024 threads](results/results.png)
![Growth Chart for SpeedUp and GFLOPS](results/growth.png)

## Resources
I was able to implement this largely inspired by https://www.youtube.com/watch?v=GetaI7KhbzM