cmake_minimum_required(VERSION 3.16)
project(CudaBook LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

add_executable(HeatTransfer heat_transfer.cu)
add_executable(HeatTransferTexture1D heat_transfer_texture_1d.cu)
add_executable(HeatTransferTexture2D heat_transfer_texture_2d.cu)

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
include_directories(${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})

target_link_libraries(HeatTransfer ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_link_libraries(HeatTransferTexture1D ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_link_libraries(HeatTransferTexture2D ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
