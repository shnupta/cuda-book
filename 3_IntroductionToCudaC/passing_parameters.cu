#include <stdio.h>

// This is a kernel and runs on the device
__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main (void) {
    int c;
    int *dev_c; // Pointer to the address on the device which holds are data
    // DO NOT dereference this pointer in host code. It is only valid when running on device code.

    cudaMalloc((void**)&dev_c, sizeof(int));

    add<<<1,1>>>(2, 7, dev_c);

    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("2 + 7 = %d\n", c);
    cudaFree(dev_c);

    return 0;
}
