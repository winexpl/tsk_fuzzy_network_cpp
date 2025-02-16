#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
int a() {
    std::string filename = "resource/irises.csv";
    std::ifstream file{filename};
    if(!file.is_open()) {
        throw std::runtime_error("file not found or not permission");
    }
    std::string line;
    file
    file >> line;
    std::cout << line;
    return 0;
}