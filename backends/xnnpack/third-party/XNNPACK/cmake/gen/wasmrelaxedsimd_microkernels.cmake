# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for wasmrelaxedsimd
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(ALL_WASMRELAXEDSIMD_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int16-u8.c
  src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int16-u16.c
  src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int16-u24.c
  src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int16-u32.c
  src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int32-u8.c
  src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int32-u16.c
  src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int32-u24.c
  src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int32-u32.c
  src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmrelaxedsimd-acc2.c
  src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmrelaxedsimd.c
  src/f32-dwconv/gen/f32-dwconv-3p4c-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmrelaxedsimd-acc2.c
  src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmrelaxedsimd.c
  src/f32-dwconv/gen/f32-dwconv-3p8c-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmrelaxedsimd-acc2.c
  src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmrelaxedsimd.c
  src/f32-dwconv/gen/f32-dwconv-4p4c-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmrelaxedsimd-acc2.c
  src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmrelaxedsimd.c
  src/f32-dwconv/gen/f32-dwconv-4p8c-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmrelaxedsimd-acc2.c
  src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmrelaxedsimd.c
  src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmrelaxedsimd-acc2.c
  src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmrelaxedsimd.c
  src/f32-dwconv/gen/f32-dwconv-9p4c-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmrelaxedsimd-acc2.c
  src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmrelaxedsimd.c
  src/f32-dwconv/gen/f32-dwconv-9p8c-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmrelaxedsimd-acc2.c
  src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmrelaxedsimd.c
  src/f32-dwconv/gen/f32-dwconv-25p4c-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmrelaxedsimd-acc2.c
  src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmrelaxedsimd-fma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmrelaxedsimd-fma.c
  src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmrelaxedsimd.c
  src/f32-dwconv/gen/f32-dwconv-25p8c-wasmrelaxedsimd-fma.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-u8.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-u16.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-u24.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-u32.c
  src/f32-gemm/gen/f32-gemm-1x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-1x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-1x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemm/gen/f32-gemm-1x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemm/gen/f32-gemm-1x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-1x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-1x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-1x8-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-1x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-1x8s4-minmax-wasmrelaxedsimd.c
  src/f32-gemm/gen/f32-gemm-1x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-1x8s4-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-3x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-3x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-3x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemm/gen/f32-gemm-3x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemm/gen/f32-gemm-3x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-3x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-3x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-3x8-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-3x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-3x8s4-minmax-wasmrelaxedsimd.c
  src/f32-gemm/gen/f32-gemm-3x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-3x8s4-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-4x2c4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-4x2c4-minmax-wasmrelaxedsimd.c
  src/f32-gemm/gen/f32-gemm-4x2c4-relu-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-4x2c4-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-4x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-4x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-4x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemm/gen/f32-gemm-4x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemm/gen/f32-gemm-4x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-4x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-4x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-4x8-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-4x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-4x8s4-minmax-wasmrelaxedsimd.c
  src/f32-gemm/gen/f32-gemm-4x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-4x8s4-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-5x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-5x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-5x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemm/gen/f32-gemm-5x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemm/gen/f32-gemm-5x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-5x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-5x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-5x8-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-5x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-5x8s4-minmax-wasmrelaxedsimd.c
  src/f32-gemm/gen/f32-gemm-5x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-5x8s4-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-6x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-6x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-6x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemm/gen/f32-gemm-6x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemm/gen/f32-gemm-6x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-6x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-6x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemm/gen/f32-gemm-6x8-wasmrelaxedsimd-fma-splat.c
  src/f32-gemm/gen/f32-gemm-6x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-6x8s4-minmax-wasmrelaxedsimd.c
  src/f32-gemm/gen/f32-gemm-6x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-gemm/gen/f32-gemm-6x8s4-wasmrelaxedsimd-fma.c
  src/f32-gemminc/gen/f32-gemminc-1x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-1x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemminc/gen/f32-gemminc-1x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-1x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemminc/gen/f32-gemminc-1x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemminc/gen/f32-gemminc-1x8s4-minmax-wasmrelaxedsimd.c
  src/f32-gemminc/gen/f32-gemminc-3x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-3x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemminc/gen/f32-gemminc-3x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-3x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemminc/gen/f32-gemminc-3x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemminc/gen/f32-gemminc-3x8s4-minmax-wasmrelaxedsimd.c
  src/f32-gemminc/gen/f32-gemminc-4x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-4x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemminc/gen/f32-gemminc-4x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-4x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemminc/gen/f32-gemminc-4x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemminc/gen/f32-gemminc-4x8s4-minmax-wasmrelaxedsimd.c
  src/f32-gemminc/gen/f32-gemminc-5x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-5x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemminc/gen/f32-gemminc-5x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-5x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemminc/gen/f32-gemminc-5x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemminc/gen/f32-gemminc-5x8s4-minmax-wasmrelaxedsimd.c
  src/f32-gemminc/gen/f32-gemminc-6x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-6x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-gemminc/gen/f32-gemminc-6x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-gemminc/gen/f32-gemminc-6x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-gemminc/gen/f32-gemminc-6x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-gemminc/gen/f32-gemminc-6x8s4-minmax-wasmrelaxedsimd.c
  src/f32-ibilinear/gen/f32-ibilinear-wasmrelaxedsimd-c4.c
  src/f32-ibilinear/gen/f32-ibilinear-wasmrelaxedsimd-c8.c
  src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-igemm/gen/f32-igemm-1x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-1x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-1x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-1x8-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-1x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-1x8s4-minmax-wasmrelaxedsimd.c
  src/f32-igemm/gen/f32-igemm-1x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-1x8s4-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-igemm/gen/f32-igemm-3x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-3x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-3x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-3x8-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-3x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-3x8s4-minmax-wasmrelaxedsimd.c
  src/f32-igemm/gen/f32-igemm-3x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-3x8s4-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-4x2c4-minmax-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-4x2c4-minmax-wasmrelaxedsimd.c
  src/f32-igemm/gen/f32-igemm-4x2c4-relu-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-4x2c4-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-igemm/gen/f32-igemm-4x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-4x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-4x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-4x8-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-4x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-4x8s4-minmax-wasmrelaxedsimd.c
  src/f32-igemm/gen/f32-igemm-4x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-4x8s4-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-igemm/gen/f32-igemm-5x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-5x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-5x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-5x8-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-5x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-5x8s4-minmax-wasmrelaxedsimd.c
  src/f32-igemm/gen/f32-igemm-5x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-5x8s4-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-igemm/gen/f32-igemm-6x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-6x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-6x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-igemm/gen/f32-igemm-6x8-wasmrelaxedsimd-fma-splat.c
  src/f32-igemm/gen/f32-igemm-6x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-6x8s4-minmax-wasmrelaxedsimd.c
  src/f32-igemm/gen/f32-igemm-6x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-igemm/gen/f32-igemm-6x8s4-wasmrelaxedsimd-fma.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-1x4.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-1x8.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-1x16.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-2x4.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-2x8.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-2x16.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-4x4.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-4x8.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-4x16.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-1x4.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-1x8.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-1x16.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-2x4.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-2x8.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-2x16.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-4x4.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-4x8.c
  src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-4x16.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8s4-minmax-wasmrelaxedsimd.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8s4-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8s4-minmax-wasmrelaxedsimd.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x8s4-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x2c4-minmax-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x2c4-minmax-wasmrelaxedsimd.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x2c4-relu-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x2c4-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8s4-minmax-wasmrelaxedsimd.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8s4-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8s4-minmax-wasmrelaxedsimd.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8s4-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-wasmrelaxedsimd-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-wasmrelaxedsimd-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-relu-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-relu-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-wasmrelaxedsimd-fma-loadsplat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-wasmrelaxedsimd-fma-splat.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8s4-minmax-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8s4-minmax-wasmrelaxedsimd.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8s4-relu-wasmrelaxedsimd-fma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8s4-wasmrelaxedsimd-fma.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u4.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u8-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u8.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u12-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u12-acc3.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u12.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u16-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u16-acc4.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u16.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u20-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u20-acc5.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u20.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm-pipelined.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm-x2.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm-x4.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86-pipelined.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86-x2.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86-x4.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm-pipelined.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm-x2.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm-x4.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86-pipelined.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86-x2.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86-x4.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm-pipelined.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm-x2.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm-x4.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86-pipelined.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86-x2.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86-x4.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm-pipelined.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm-x2.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm-x4.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86-pipelined.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86-x2.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86-x4.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u4.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u8.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u12.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u16.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u20.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u24.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u4.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u8.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u12.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u16.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u20.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u24.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u4.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u8.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u12.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u16.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u20.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u24.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u4.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u8.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u12.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u16.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u20.c
  src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u24.c
  src/f32-vlrelu/gen/f32-vlrelu-wasmrelaxedsimd-iminmax-u4.c
  src/f32-vlrelu/gen/f32-vlrelu-wasmrelaxedsimd-iminmax-u8.c
  src/f32-vlrelu/gen/f32-vlrelu-wasmrelaxedsimd-laneselect-u4.c
  src/f32-vlrelu/gen/f32-vlrelu-wasmrelaxedsimd-laneselect-u8.c
  src/f32-vmulcaddc/gen/f32-vmulcaddc-c4-minmax-wasmrelaxedsimd-2x.c
  src/f32-vmulcaddc/gen/f32-vmulcaddc-c4-minmax-wasmrelaxedsimd-fma-2x.c
  src/f32-vmulcaddc/gen/f32-vmulcaddc-c8-minmax-wasmrelaxedsimd-2x.c
  src/f32-vmulcaddc/gen/f32-vmulcaddc-c8-minmax-wasmrelaxedsimd-fma-2x.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u20.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u24.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u20.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u24.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u20.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u24.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u20.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u24.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u20.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u24.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u20.c
  src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u24.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c16-minmax-wasmusdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c16-minmax-wasmusdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c16-minmax-wasmusdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c16-minmax-wasmusdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c16-minmax-wasmusdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c16-minmax-wasmusdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c16-minmax-wasmusdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c16-minmax-wasmusdot.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c16-minmax-wasmsdot.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c16-minmax-wasmsdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c16-minmax-fp32-wasmusdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c16-minmax-fp32-wasmsdot.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c16-minmax-fp32-wasmusdot.c
  src/qs8-vcvt/gen/qs8-vcvt-wasmrelaxedsimd-u8.c
  src/qs8-vcvt/gen/qs8-vcvt-wasmrelaxedsimd-u16.c
  src/qs8-vcvt/gen/qs8-vcvt-wasmrelaxedsimd-u32.c
  src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-arm-u16.c
  src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-arm-u32.c
  src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-x86-u8.c
  src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-x86-u16.c
  src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-x86-u32.c
  src/qu8-vcvt/gen/qu8-vcvt-wasmrelaxedsimd-u8.c
  src/qu8-vcvt/gen/qu8-vcvt-wasmrelaxedsimd-u16.c
  src/qu8-vcvt/gen/qu8-vcvt-wasmrelaxedsimd-u32.c
  src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-arm-u16.c
  src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-arm-u32.c
  src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-x86-u8.c
  src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-x86-u16.c
  src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-x86-u32.c
  src/x8-lut/gen/x8-lut-wasmpshufb-u16.c
  src/x8-lut/gen/x8-lut-wasmpshufb-u32.c
  src/x8-lut/gen/x8-lut-wasmpshufb-u48.c
  src/x8-lut/gen/x8-lut-wasmpshufb-u64.c)