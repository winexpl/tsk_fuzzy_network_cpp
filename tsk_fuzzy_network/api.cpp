#include "api.h"
#include "logger.h"

Dataset dataset;
int a = 0;

tsk::TSK* createTSK(int n, int m)
{
    return new tsk::TSK(n, m);
}

void deleteTSK(tsk::TSK* ptr)
{
    Logger::getInstance().logInfo("model deleted");
    delete ptr;
}

int test() {
    a++;
    return a;
}

tsk::TSK* learning(tsk::TSK* tsk, Callback callback) {
    callback(2);
    return tsk;
}

Dataset *readFromCsv(const char* filenameJava) {
    std::string filename(filenameJava);
    Dataset *created = new Dataset(Dataset::readFromCsv(filename));
    std::cout << created << std::endl;
    return created;
}

