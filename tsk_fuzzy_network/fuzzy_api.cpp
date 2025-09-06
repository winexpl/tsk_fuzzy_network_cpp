#include "fuzzy_api.h"
#include "tsk_fuzzy_network/tsk.h"
#include "tsk_fuzzy_network/learning_algorithms.h"
#include "tsk_fuzzy_network/c_means.h"
#include <boost/multi_array.hpp>

#include <sstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

void* create_tsk_model(int n, int m) {
    return new tsk::TSK(n, m);
}

void free_tsk_model(void* tsk_row) {
    delete static_cast<tsk::TSK*>(tsk_row);
}

void free_dataset(Dataset* dataset_row) {
    delete dataset_row;
}

void free_hybrid_algorithm(void* halg_raw) {
    delete static_cast<learning::HybridAlgorithm*>(halg_raw);
}

void* create_hybrid_alg(void* tsk_row, void* dataset_row) {
    auto tsk = static_cast<tsk::TSK*>(tsk_row);
    auto dataset = *static_cast<Dataset*>(dataset_row);

    tsk::CMeans cmeans(dataset.getX().shape()[0], 0, 0.0001);
    cmeans.fit(dataset.getX());
    std::vector<double> c = cmeans.getCentroids();
    std::vector<double> sigma = cmeans.getSigma();

    tsk->setC(c);
    tsk->setSigma(sigma);
    
    return new learning::HybridAlgorithm(tsk, dataset);
}

Dataset *load_dataset(char* path)
{
    std::string _path(path);
    Dataset dataset = Dataset::readFromCsv(_path);
    Dataset* newDataset = new Dataset(dataset);
    std::cout << newDataset->dim << "\n";
    return newDataset;
}

Dataset* create_dataset(void* x_raw, int x_rows, int x_cols, void* d_raw, int d_len, int classCount)
{
    boost::multi_array<double, 2> x(boost::extents[x_rows][x_cols]);

    double* x_data = static_cast<double*>(x_raw);
    double* d_data = static_cast<double*>(d_raw);

    std::copy(x_data, x_data + x_rows * x_cols, x.data());

    std::vector<double> d(d_data, d_data + d_len);
    return new Dataset(x, d, x_rows, classCount, x_cols);
}

void learning_tsk(void *halg_raw)
{
    learning::HybridAlgorithm *halg = static_cast<learning::HybridAlgorithm*>(halg_raw);
    halg->learning(halg->dataset.getCountVectors(), 1, 5, learning::TrainingConfig{});
}

std::vector<double> tsk_predict(void* model, double* input, int input_rows, int input_cols, int* output_size) 
{
    tsk::TSK* tsk = static_cast<tsk::TSK*>(model);
    boost::multi_array<double, 2> x(boost::extents[input_rows][input_cols]);
    std::copy(input, input + input_rows * input_cols, x.data());
    auto result = tsk->predict(x);
    *output_size = result.size();
    double* out = new double[result.size()];
    std::copy(result.begin(), result.end(), out);
    return result;  // Java должна освободить память через release_array
}

void release_array(double* arr) {
    delete[] arr;
}

std::string serialize_tsk(void *tsk_raw) {
    tsk::TSK *tsk = static_cast<tsk::TSK*>(tsk_raw);
    std::ostringstream oss;
    boost::archive::text_oarchive oa(oss);
    oa << *tsk;
    return oss.str();
}

void *deserialize_tsk(std::string tsk_serialized) {
    std::istringstream iss(tsk_serialized);
    boost::archive::text_iarchive ia(iss);
    tsk::TSK* tsk = new tsk::TSK();
    ia >> *tsk;
    return tsk;
}

std::string serialize_dataset(Dataset* dataset_raw) {
    std::cout << "1111\n";
    // Dataset *dataset = dataset_raw;
    std::cout << dataset_raw->getClassCount() << "\n";
    std::cout << dataset_raw->getX().shape()[0] << " " << dataset_raw->getX().shape()[1] << "\n";
    std::ostringstream oss;
    std::cout << "3333\n";
    boost::archive::text_oarchive oa(oss);
    std::cout << "4444\n";
    oa << *dataset_raw;
    std::cout << "5555\n";
    return oss.str();
}

Dataset* deserialize_dataset(const char* data) {
    std::istringstream iss(data);
    boost::archive::text_iarchive ia(iss);
    Dataset* dataset = new Dataset();
    ia >> *dataset;
    return dataset;
}