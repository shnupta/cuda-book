#include <iostream>

int main() {
    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++) {
	cudaGetDeviceProperties(&prop, i);
	std::cout << "   --- General Information for device " << i << " ---\n";
	std::cout << "Name: " << prop.name << std::endl;
	std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
	std::cout << "Clock rate: " << prop.clockRate << std::endl;
	std::cout << "Device copy overlap: " << (prop.deviceOverlap ? "Enabled\n" : "Disabled\n");
	std::cout << "Kernel execution timeout: " << (prop.kernelExecTimeoutEnabled ? "Enabled\n" : "Disabled\n");

	std::cout << "   --- Memory Information for device " << i << " ---\n";
	std::cout << "Total glob mem: " << prop.totalGlobalMem << std::endl;
	std::cout << "Total constant mem: " << prop.totalConstMem << std::endl;
	std::cout << "Max mem pitch: " << prop.memPitch << std::endl;
	std::cout << "Texture alignment: " << prop.textureAlignment << std::endl;

	std::cout << "   --- MP Information for device " << i << " ---\n";
	std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
	std::cout << "Shared mem per mp: " << prop.sharedMemPerBlock << std::endl;
	std::cout << "Registers per mp: " << prop.regsPerBlock << std::endl;
	std::cout << "Threads in ward: " << prop.warpSize << std::endl;
	std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
	std::cout << "Max thread dimensions: (" << prop.maxThreadsDim[0] << ", " 
	    << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
	std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", " 
	    << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";

	std::cout << std::endl;


    }
    return 0;
}
