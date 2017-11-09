# Unix commands.
PYTHON := python
NVCC := /usr/local/cuda-8.0/bin/nvcc
NVCC_COMPILE := $(NVCC) -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -c -o
RM_RF := rm -rf

# Library compilation rules.
NVCC_FLAGS := -x cu -Xcompiler -fPIC -shared

# File structure.
BUILD_DIR := build
INCLUDE_DIRS := include
TORCH_FFI_BUILD := build_ffi.py
MATHUTIL_KERNEL := $(BUILD_DIR)/shiftnet_cuda_kernels.so
TORCH_FFI_TARGET := $(BUILD_DIR)/shiftnet_cuda/_shiftnet_cuda.so

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$d)

all: $(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET): $(MATHUTIL_KERNEL) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)

$(BUILD_DIR)/%.so: src/%.cu
	@ mkdir -p $(BUILD_DIR)
	# Separate cpp shared library that will be loaded to the extern C ffi
	$(NVCC_COMPILE) $@ $? $(NVCC_FLAGS) $(INCLUDE_FLAGS)

clean:
	$(RM_RF) $(BUILD_DIR) shiftnet_cuda/__init__.py shiftnet_cuda/_shiftnet_cuda.so *.pyc
