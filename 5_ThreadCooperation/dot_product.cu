#include <stdio.h>

#define imin(a, b) (a < b ? a : b)
#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {
  __shared__ float cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  // Set the cache value
  cache[cacheIndex] = temp;

  // Require all threads in this block to reach this point before any single one can continue
  __syncthreads();

  // For reductions, threadsPerBlock must be a power of 2
  int i = blockDim.x / 2; // Number of threads in this block / 2 (each thread handles 2 values)
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads(); // Wait for all threads again before summing the next i / 2 values
    i /= 2;
  }

  // Each block has a single number left now, copy this to the global c
  if (cacheIndex == 0) {
    c[blockIdx.x] = cache[0];
  }
}

int main() {
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;

  // Allocate memory on the CPU side
  a = (float *)malloc(N * sizeof(float));
  b = (float *)malloc(N * sizeof(float));
  partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

  // Allocate memory on the GPU side (I know I should be error checking here like with all previous examples)
  cudaMalloc((void **)&dev_a, N * sizeof(float));
  cudaMalloc((void **)&dev_b, N * sizeof(float));
  cudaMalloc((void **)&dev_partial_c, blocksPerGrid * sizeof(float));

  // Fill the host memory with the data
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  // Copy the arrays to the GPU
  cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

  dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

  // Copy partial sums from device to host memory
  cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

  c = 0;
  for (int i = 0; i < blocksPerGrid; i++) {
    c += partial_c[i];
  }

  printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float) (N - 1)));

  // Free GPU memory
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_c);

  free(a);
  free(b);
  free(partial_c);
}
