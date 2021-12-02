#include <iostream>

const int N = 33 * 1024;

__global__ void add(int *a, int *b, int *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N) {
    c[tid] = a[tid] + b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

int main() {
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  // Allocate memory on the GPU
  cudaMalloc((void**)&dev_a, N * sizeof(int));
  cudaMalloc((void**)&dev_b, N * sizeof(int));
  cudaMalloc((void**)&dev_c, N * sizeof(int));

  // Fill the arrays a and b on the CPU
  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = i * 1;
  }

  // Copy the arrays a and b to the GPU
  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  add<<<128,128>>>(dev_a, dev_b, dev_c);

  // Copy the array c back from GPU to the CPU
  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  // Display the results
  for (int i = 0; i < N; ++i) {
    std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
  }

  // Free the GPU allocated memory
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}
