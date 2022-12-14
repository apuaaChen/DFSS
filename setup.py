from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

WDIR = os.getcwd()

setup(
    name='dfss',
    version='0.0.1',
    description='Custom library for Dynamic Sparse Self-Attention for pytorch',
    author='Zhaodong Chen',
    author_email='chenzd15thu@ucsb.edu',
    package_dir={'':"src"},
    packages=['pydfss'],
    ext_modules=[
        CUDAExtension('dfss.meta', 
                      ['src/cuda/meta.cpp', 'src/cuda/meta_kernel.cu'],
                      extra_compile_args={'cxx':['-lineinfo'], 'nvcc':['-arch=sm_80', '--ptxas-options=-v', '-lineinfo', '-lcublass', '-use_fast_math']},
                      include_dirs=[WDIR+'/thirdparty/cutlass/include', WDIR+'/thirdparty/cutlass/tools/util/include', WDIR+'/thirdparty/cutlass/examples/common']),
        CUDAExtension('dfss.spmm', 
                      ['src/cuda/spmm.cpp', 'src/cuda/spmm_kernel.cu'],
                      extra_cuda_cflags=['-lineinfo'],
                      extra_compile_args={'cxx':['-lineinfo'], 'nvcc':['-arch=sm_80', '--ptxas-options=-v', '-lineinfo', '-lcublass', '-use_fast_math']},
                      include_dirs=[WDIR+'/thirdparty/cutlass/include', WDIR+'/thirdparty/cutlass/tools/util/include', WDIR+'/thirdparty/cutlass/examples/common']),
        CUDAExtension('dfss.sddmm', 
                      ['src/cuda/sddmm.cpp', 'src/cuda/sddmm_kernel.cu'],
                      extra_cuda_cflags=['-lineinfo'],
                      extra_compile_args={'cxx':['-lineinfo'], 'nvcc':['-arch=sm_80', '--ptxas-options=-v', '-lineinfo', '-lcublass', '-use_fast_math']},
                      include_dirs=[WDIR+'/thirdparty/cutlass/include', WDIR+'/thirdparty/cutlass/tools/util/include', WDIR+'/thirdparty/cutlass/examples/common']),
        ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)