#pragma once

#include <vector>
#include <utility>
#include <string>

#ifdef __cplusplus
extern "C"
{
#endif
    void *create_tsk_model(int n, int m);
    void free_tsk_model(void *model);

    std::vector<double> tsk_predict(void *model, double *input, int input_rows, int input_cols, int *output_size);
    std::pair<double, double> tsk_evaluate(void *model, double *input, int input_size);

    void* create_hybrid_alg(void* tsk_row, void* dataset_row);
    void learning_tsk(void *halg_raw);
    void free_hybrid_algorithm(void* halg_raw);

    void *create_dataset(void *x, int x_rows, int x_cols, void *d, int d_len, int classCount);
    void *load_dataset(char *path);
    void free_dataset(void *dataset);

    void *deserialize_tsk(std::string tsk_serialized);
    std::string serialize_tsk(void *tsk_raw);

    std::string serialize_dataset(void* dataset);
    void* deserialize_dataset(const char* data);

#ifdef __cplusplus
};
#endif