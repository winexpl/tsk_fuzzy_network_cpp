#pragma once
#include <iostream>
#include <vector>
#include <boost/multi_array.hpp>

// std::ostream& operator<<(std::ostream& os, std::vector<double>& x) {
//     for(int i = 0; i < x.size(); i++) {
//         os << "x[" << i << "] = " << x[i] << "\n";
//     }
//     os.flush();
//     return os;
// }

// std::ostream& operator<<(std::ostream& os, boost::multi_array<double,2>& x) {
//     for(int i = 0; i < x.shape()[1]; i++) {
//         for(int j = 0; j < x.shape()[0]; j++) {
//             os << "x[" << i << "][" << j << "] = " << x[i][j] << "\n";
//         }
//     }
//     os.flush();
//     return os;
// }
// 