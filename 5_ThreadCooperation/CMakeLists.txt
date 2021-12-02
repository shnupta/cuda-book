cmake_minimum_required(VERSION 3.16)
project(CudaBook LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

add_executable(SummingVectorsThreads summing_vectors_threads.cu)
add_executable(SummingVectorsBlocksThreads summing_vectors_blocks_threads.cu)
add_executable(GPURipple gpu_ripple.cu ../common/cpu_anim.h)
add_executable(DotProduct dot_product.cu)
add_executable(SharedBitmap shared_bitmap.cu)

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
include_directories(${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})

target_link_libraries(GPURipple ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_link_libraries(SharedBitmap ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
