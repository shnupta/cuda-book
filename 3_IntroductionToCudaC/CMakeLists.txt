project(CudaBook LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

add_executable(HelloWorld hello_world.cu)
add_executable(PassingParameters passing_parameters.cu)
add_executable(DeviceProperties device_properties.cu)
add_executable(ChooseDevice choose_device.cu)
