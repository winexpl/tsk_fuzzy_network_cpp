#include "tsk_fuzzy_network/layers.h"
#include "tsk_fuzzy_network/tsk.h"
#include "csv_reader.h"
#include <iostream>

std::ostream& operator<<(std::ostream& os, std::vector<double>&& x) {
    for(int i = 0; i < x.size(); i++) {
        os << "x[" << i << "] = " << x[i] << "\n";
    }
    os.flush();
    return os;
}

int main(int argc, char* argv[]) {
    tsk::TSK tsk(3,1);
    std::vector<double> x{3,1,2};
    std::cout << tsk.predict(x) << std::endl;
    a();
}
