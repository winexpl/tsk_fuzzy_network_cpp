#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

void convert_irises ();
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

std::ostringstream join(const std::vector<std::string>& tokens, char delimiter) {
    std::ostringstream new_line;
    for (size_t i = 0; i < tokens.size(); ++i) {
        new_line << tokens[i];
        if (i < tokens.size() - 1) {
            new_line << delimiter;
        }
    }
    return new_line;
}

int main() {
    convert_irises();
}

void convert_irises () {
    std::string path_old_file = "./resource/old_irises.csv";
    std::string path_new_file = "./resource/new_irises.csv";
    std::ifstream old_file{path_old_file};
    std::ofstream new_file{path_new_file, std::ios::trunc};
    std::cout << std::boolalpha << new_file.is_open() << std::endl;
    
    std::string line;
    old_file >> line;
    new_file << line;
    new_file << "\n";
    while(old_file.good()) {
        
        old_file >> line;
        std::istringstream old_line {line};
        char delimiter = ',';
        std::vector<std::string> tokens = split(line, delimiter);
        std::string out = tokens.back();
        tokens.pop_back();
        std::ostringstream new_line = join(tokens, delimiter);

        if(out == "Iris-setosa") {
            new_line << ",0\n";
        } else if(out == "Iris-versicolor") {
            new_line << ",0.5\n";
        } else if(out == "Iris-virginica") {
            new_line << ",1\n";
        }
        new_file << new_line.str();
    }
}