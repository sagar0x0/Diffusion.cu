cmake_minimum_required(VERSION 3.18)
project(flash_attention_module LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Python first
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Find PyTorch
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Then find pybind11
find_package(pybind11 REQUIRED)

# pybind module name flash_atten 
pybind11_add_module(flash_atten_kernel cuda_kernel/flash_atten.cu)
pybind11_add_module(groupNorm_kernel cuda_kernel/groupnorm.cu)
pybind11_add_module(Conv2d_cuda_kernel cuda_kernel/conv2d_k3.cu)

# target GPU architectures
# targets P100 for now : 6.0
# Check your GPU's compute capability.
set_property(TARGET flash_atten_kernel PROPERTY CUDA_ARCHITECTURES "60")
set_property(TARGET groupNorm_kernel PROPERTY CUDA_ARCHITECTURES "60")
set_property(TARGET Conv2d_cuda_kernel PROPERTY CUDA_ARCHITECTURES "60")

# Add include directories and link libraries
target_include_directories(flash_atten_kernel PRIVATE ${pybind11_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_include_directories(groupNorm_kernel PRIVATE ${pybind11_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_include_directories(Conv2d_cuda_kernel PRIVATE ${pybind11_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

target_link_libraries(flash_atten_kernel PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(groupNorm_kernel PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(Conv2d_cuda_kernel PRIVATE ${TORCH_LIBRARIES})

# Add CUDA specific flags
target_compile_options(flash_atten_kernel PRIVATE $<BUILD_INTERFACE:${TORCH_CXX_FLAGS}>)
target_compile_options(groupNorm_kernel PRIVATE $<BUILD_INTERFACE:${TORCH_CXX_FLAGS}>)
target_compile_options(Conv2d_cuda_kernel PRIVATE $<BUILD_INTERFACE:${TORCH_CXX_FLAGS}>)

target_compile_definitions(flash_atten_kernel PRIVATE $<BUILD_INTERFACE:${TORCH_COMPILE_DEFINITIONS}>)
target_compile_definitions(groupNorm_kernel PRIVATE $<BUILD_INTERFACE:${TORCH_COMPILE_DEFINITIONS}>)
target_compile_definitions(Conv2d_cuda_kernel PRIVATE $<BUILD_INTERFACE:${TORCH_COMPILE_DEFINITIONS}>)

# Add this to your CMakeLists.txt before pybind11_add_module
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Override the default naming convention
set_target_properties(flash_atten_kernel PROPERTIES SUFFIX ".so")
set_target_properties(groupNorm_kernel PROPERTIES SUFFIX ".so")
set_target_properties(Conv2d_cuda_kernel PROPERTIES SUFFIX ".so")
