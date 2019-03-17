#!/bin/bash
INSTALL_PREFIX=/users/profs/2017/francois.trahay/soft/install

# HWLOC stuff
export HWLOC_ROOT=$INSTALL_PREFIX/hwloc
PATH=$HWLOC_ROOT/bin/:$PATH
MANPATH=$MANPATH:$HWLOC_ROOT/share/man
LD_LIBRARY_PATH=$HWLOC_ROOT/lib:$LD_LIBRARY_PATH
PKG_CONFIG_PATH=$HWLOC_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH

# CUDA stuff
export CUDA_ROOT=/usr/local/cuda-10.0
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_ROOT}/lib64
PATH=${CUDA_ROOT}/bin/:${CUDA_ROOT}/open64/bin:$PATH

# MPI stuff
export MPI_ROOT=$INSTALL_PREFIX/openmpi/
PATH=${MPI_ROOT}/bin:$PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MPI_ROOT}/lib
MANPATH=$MANPATH:$MPI_ROOT/share/man

# EZTrace
export EZTRACE_ROOT=$INSTALL_PREFIX/eztrace/
PATH=$EZTRACE_ROOT/bin:$PATH

# ViTE
export VITE_ROOT=$INSTALL_PREFIX/vite
export PATH=$VITE_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VITE_ROOT/lib
export QT_QPA_PLATFORM_PLUGIN_PATH=$VITE_ROOT/plugins

export LD_LIBRARY_PATH
export PATH
export MANPATH
export PKG_CONFIG_PATH
