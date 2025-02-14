#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include <iostream>

std::ostream& operator<<(std::ostream& os, std::vector<double>&& x) {
    for(int i = 0; i < x.size(); i++) {
        os << "x[" << i << "] = " << x[i] << "\n";
    }
    os.flush();
    return os;
}

int main(int argc, char* argv[]) {
    tsk::TSK tsk(1,1);
    std::vector<double> x{1};
    
    std::cout << tsk.predict(x) << std::endl;
}