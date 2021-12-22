cmake_minimum_required(VERSION 3.16)
project(CudaBook LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)
# Would need this but the toolkit is much older now so the lowest compute capability is way higher 
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=compute_12")

add_executable(CPUHistogram cpu_histogram.cu)
add_executable(GPUHistogram gpu_histogram.cu)
add_executable(GPUHistogramShared gpu_histogram_shared.cu)

# find_package(GLUT REQUIRED)
# find_package(OpenGL REQUIRED)
# include_directories(${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})

# target_link_libraries(HeatTransfer ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
