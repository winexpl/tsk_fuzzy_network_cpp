#ifndef C_MEANS_H
#define C_MEANS_H

#include <vector>
#include <cmath>
#include <limits>
#include <boost/multi_array.hpp>

namespace tsk {

class CMeans {
public:
    CMeans(unsigned int clusters, double fuzziness, double epsilon)
        : m_clusters(clusters), m_fuzziness(fuzziness), m_epsilon(epsilon) {}

    void fit(const boost::multi_array<double, 2>& data, unsigned int maxIterations = 100);

    const std::vector<double> getCentroids() const
    {
        std::vector<double> centroids1d;
        for (const auto& row : m_centroids) {
            std::copy(row.begin(), row.end(), std::back_inserter(centroids1d));
        }
        return centroids1d;
    }
    const std::vector<double> getSigma() const
    {
        std::vector<double> sigma1d;
        for (const auto& row : m_sigma) {
            std::copy(row.begin(), row.end(), std::back_inserter(sigma1d));
        }
        return sigma1d;
    }
    const std::vector<std::vector<double>>& getMembershipMatrix() const { return m_membershipMatrix; }

private:
    unsigned int m_clusters;
    double m_fuzziness;
    double m_epsilon;

    std::vector<std::vector<double>> m_centroids;
    std::vector<std::vector<double>> m_sigma;
    std::vector<std::vector<double>> m_membershipMatrix;

    void initializeMembershipMatrix(size_t dataSize);
    void updateCentroids(const boost::multi_array<double, 2> &data);
    void updateMembershipMatrix(const boost::multi_array<double, 2>& data);
    double calculateDistance(const double& point1, const double& point2) const;
};

} // namespace tsk

#endif // C_MEANS_H