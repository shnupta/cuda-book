#include <stdio.h>

// __global__ indicates to the compiler we intend to run this function on the GPU
__global__ void kernel(void) {}

int main(void) {
  kernel<<<1,1>>>();
  printf("Hello, World!\n");
  return 0;
}
