############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
############################################################
open_project cnn_general
set_top cnn_general
add_files cnn.cpp
add_files cnn.h
add_files cnn_impl.h
add_files -tb cnn_general/cnn_general_test.cpp
open_solution "solution1"
set_part {xc7z020clg484-1} -tool vivado
create_clock -period 10 -name default
#source "./cnn_general/solution1/directives.tcl"
csim_design -compiler gcc -setup
csynth_design
cosim_design
export_design -format ip_catalog
