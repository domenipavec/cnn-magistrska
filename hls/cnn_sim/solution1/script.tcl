############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
############################################################
open_project cnn_sim
set_top cnn_sim
add_files cnn.h
add_files cnn_class.h
add_files cnn_impl.h
add_files cnn_sim/cnn_sim.cpp
add_files cnn_sim/cnn_sim.h
add_files cnn_streamw.h
add_files -tb cnn_sim/cnn_sim_test.cpp
open_solution "solution1"
set_part {xc7z020clg484-1} -tool vivado
create_clock -period 10 -name default
#source "./cnn_sim/solution1/directives.tcl"
csim_design -clean -compiler gcc -setup
csynth_design
cosim_design
export_design -format ip_catalog
