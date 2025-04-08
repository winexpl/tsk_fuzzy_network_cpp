#include "tsk_fuzzy_network/c_means.h"

namespace tsk {

    void CMeans::initializeMembershipMatrix(size_t dataSize) {
        m_membershipMatrix.resize(dataSize, std::vector<double>(m_clusters));
        for (size_t i = 0; i < dataSize; ++i) {
            double sum = 0.0;
            for (unsigned int j = 0; j < m_clusters; ++j) {
                m_membershipMatrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
                sum += m_membershipMatrix[i][j];
            }
            for (unsigned int j = 0; j < m_clusters; ++j) {
                m_membershipMatrix[i][j] /= sum;
            }
        }
    }
    
    void CMeans::updateCentroids(const boost::multi_array<double, 2>& data) {
        m_centroids.resize(m_clusters, std::vector<double>(data.shape()[1], 0.0));
        for (unsigned int j = 0; j < m_clusters; ++j) {
            double denominator = 0.0;
            for (size_t i = 0; i < data.shape()[0]; ++i) {
                double weight = std::pow(m_membershipMatrix[i][j], m_fuzziness);
                for (size_t k = 0; k < data.shape()[1]; ++k) {
                    m_centroids[j][k] += weight * data[i][k];
                }
                denominator += weight;
            }
            for (size_t k = 0; k < data.shape()[1]; ++k) {
                m_centroids[j][k] /= denominator;
            }
        }
    }
    
    void CMeans::updateMembershipMatrix(const boost::multi_array<double, 2>& data) {
        for (size_t i = 0; i < data.shape()[0]; ++i) {
            for (unsigned int j = 0; j < m_clusters; ++j) {
                double sum = 0.0;
                for (unsigned int k = 0; k < m_clusters; ++k) {
                    double ratio = calculateDistance(data[i][0], m_centroids[j][0]) /
                                    calculateDistance(data[i][0], m_centroids[k][0]);
                    sum += std::pow(ratio, 2.0 / (m_fuzziness - 1.0));
                }
                m_membershipMatrix[i][j] = 1.0 / sum;
            }
        }
    }
    
    double CMeans::calculateDistance(const double& point1, const double& point2) const {
        return std::sqrt(std::pow(point1 - point2, 2));
    }
    
    void CMeans::fit(const boost::multi_array<double, 2>& data, unsigned int maxIterations) {
        initializeMembershipMatrix(data.shape()[0]);
        for (unsigned int iteration = 0; iteration < maxIterations; ++iteration) {
            std::vector<std::vector<double>> oldMembershipMatrix = m_membershipMatrix;
            updateCentroids(data);
            updateMembershipMatrix(data);
    
            double maxChange = 0.0;
            for (size_t i = 0; i < m_membershipMatrix.size(); ++i) {
                for (unsigned int j = 0; j < m_clusters; ++j) {
                    maxChange = std::max(maxChange, std::abs(m_membershipMatrix[i][j] - oldMembershipMatrix[i][j]));
                }
            }
            if (maxChange < m_epsilon) {
                break;
            }
        }

        m_sigma.resize(m_clusters, std::vector<double>(m_centroids[0].size(), 0.0));
    
        for (int i = 0; i < m_clusters; ++i) {
            std::vector<double> distances(m_centroids[0].size(), 0.0); // Инициализируем временный массив
    
            // Сумма квадратов расстояний для точки
            for (const auto& point : data) { // x — исходные данные, а не c (вектор точек)
                for (size_t j = 0; j < point.size(); ++j) {
                    double diff = point[j] - m_centroids[i][j]; // Используем центр кластера c[i][j]
                    distances[j] += diff * diff;
                }
            }
    
            // Среднеквадратичное отклонение
            for (size_t j = 0; j < distances.size(); ++j) {
                m_sigma[i][j] = std::sqrt(distances[j] / m_centroids.size()); // x.size() — количество точек
            }
        }
    }
} // namespace tsk