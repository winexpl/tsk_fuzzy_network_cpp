#include "tsk_fuzzy_network/dataset.h"
#include "tsk_fuzzy_network/tsk.h"
#include <functional>

extern "C" {
    #include "tsk_fuzzy_network/dataset.h"
    #include <map>
}
extern "C" typedef void (*Callback)(double value);
extern "C" tsk::TSK *createTSK(int n, int m);
extern "C" int test();
extern "C" void deleteTSK(tsk::TSK* ptr);
extern "C" tsk::TSK* learning(tsk::TSK* tsk, Callback callback);
extern "C" Dataset *readFromCsv(const char* filename);

