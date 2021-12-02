cmake_minimum_required(VERSION 3.16)
project(CudaBook LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

add_executable(RayTracer ray_tracer.cu)

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
include_directories(${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})

target_link_libraries(RayTracer ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
