cmake_minimum_required(VERSION 3.16)
project(CudaBook LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

add_executable(SummingVectors summing_vectors.cu)
add_executable(JuliaSetCPU julia_set_cpu.cu)
add_executable(JuliaSetGPU julia_set_gpu.cu)

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
include_directories(${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})

target_link_libraries(JuliaSetCPU ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_link_libraries(JuliaSetGPU ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
