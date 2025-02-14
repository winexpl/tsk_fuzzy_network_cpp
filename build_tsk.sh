#!/bin/bash

mkdir -p ./build/tsk;
cd ./build/tsk;
cmake ../../tsk_fuzzy_network;
cmake --build .;