#ifndef METRIC_H
#define METRIC_H

#include <vector>
#include <stdexcept>
#include <math.h>

namespace metric {

class Metric {
public:

    double accuracy;
    double mse;

    Metric(double accuracy, double mse) : accuracy{accuracy}, mse{mse} {}
    Metric() = default;

    static double calculateAccuracy(const std::vector<double>& trueValues, const std::vector<double>& predictedValues, int classesCount) {
        int correct = 0;
        double neighborhood = (double)1/(2*((double)classesCount-1));
        int total = trueValues.size();
        for (size_t i = 0; i < total; ++i) {
            if ( std::abs(trueValues[i] - predictedValues[i]) < neighborhood ) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / total;
    }

    // FOR BINARY
    static double calculatePrecision(const std::vector<double>& trueValues, const std::vector<double>& predictedValues) {
        int tp = 0;
        int fp = 0;

        int total = trueValues.size();
        for (size_t i = 0; i < total; ++i) {
            if (predictedValues[i] > 0.5) {
                if (trueValues[i] > 0.5) {
                    ++tp;
                } else {
                    ++fp;
                }
            }
        }

        return tp == 0 ? 0.0 : static_cast<double>(tp) / (tp + fp);
    }

    // FOR BINARY
    static double calculateRecall(const std::vector<double>& trueValues, const std::vector<double>& predictedValues) {
        int tp = 0; // True Positives
        int fn = 0; // False Negatives

        for (size_t i = 0; i < trueValues.size(); ++i) {
            if (trueValues[i] > 0.5) {
                if (predictedValues[i] > 0.5) {
                    ++tp;
                } else {
                    ++fn;
                }
            }
        }
        return tp == 0 ? 0.0 : static_cast<double>(tp) / (tp + fn);
    }


    static double calculateF1Score(const std::vector<double>& trueValues, const std::vector<double>& predictedValues) {
        double precision = calculatePrecision(trueValues, predictedValues);
        double recall = calculateRecall(trueValues, predictedValues);

        return (precision == 0 || recall == 0) ? 0.0
            : 2.0 * (precision * recall) / (precision + recall);
    }

    static std::tuple<int, int, int, int> calculateConfusionMatrix(const std::vector<double>& trueValues, const std::vector<double>& predictedValues) {
        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (size_t i = 0; i < trueValues.size(); ++i) {
            if (trueValues[i] > 0.5 && predictedValues[i] > 0.5) {
                ++tp;
            } else if (trueValues[i] < 0.5 && predictedValues[i] < 0.5) {
                ++tn;
            } else if (trueValues[i] < 0.5 && predictedValues[i] > 0.5) {
                ++fp;
            } else if (trueValues[i] > 0.5 && predictedValues[i] < 0.5) {
                ++fn;
            }
        }

        return std::make_tuple(tp, tn, fp, fn);
    }

    static double calculateMSE(const std::vector<double>& trueValues, const std::vector<double>& predictedValues, int classesCount) {
        double sum = 0.0;
        int total = trueValues.size();

        for (int i = 0; i < total; ++i) {
            double diff = std::round(std::abs(trueValues[i] - predictedValues[i]) * (classesCount-1));
            sum += diff * diff;
        }

        return sum / total;
    }
};

}

#endif // METRIC_H