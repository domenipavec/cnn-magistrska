############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
############################################################
open_project cnn_full_layer_stack
set_top cnn_full_layer_stack
add_files cnn.cpp
add_files cnn.h
add_files cnn_impl.h
add_files -tb cnn_full_layer_stack/cnn_full_layer_stack_test.cpp
open_solution "solution1"
set_part {xc7z020clg484-1} -tool vivado
create_clock -period 10 -name default
#source "./cnn_full_layer_stack/solution1/directives.tcl"
csim_design -compiler gcc -setup
csynth_design
cosim_design
export_design -format ip_catalog
