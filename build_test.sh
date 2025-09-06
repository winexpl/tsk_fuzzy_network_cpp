#!/bin/bash

mkdir -p ./build/test;
cd ./build/test;
cmake ../../tests;
cmake --build .;