#!/bin/bash

mkdir -p ./build/main;
cd ./build/main;
cmake ../../src;
cmake --build .;