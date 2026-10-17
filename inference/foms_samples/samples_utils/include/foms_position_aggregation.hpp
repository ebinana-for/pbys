#ifndef FOMS_POSITIONS_AGGREGATION_HPP
#define FOMS_POSITIONS_AGGREGATION_HPP

#include "foms_utils.hpp"
#include <cstddef>
#include <array>
#include <cstdint>
#include <numeric>
#include <cmath>
#include "foms_positions.hpp"
/**
 * @file
 * @brief Defines classes for aggregating multiple individual position classification predictions.
 */

namespace foms {

inline constexpr size_t poseDefaultBufferSize = 20;
inline constexpr float orientationShoulderThreshold = 5.0F;
inline constexpr float distanceDashboardThreshold = 0.7F;

/// @brief Circular buffer for storing historical observations
template < size_t BufferSize > class floatBuffer {
  public:
    floatBuffer() = default;

    /// @brief Adds value to the circular buffer, filling up the oldest position
    /// @param value New value to be saved
    void push(float value) {
        head = (head + 1) % BufferSize;
        buffer[head] = value;
        numbers_elt++;
    }

    /// @brief Returns the contents of the memory buffer
    /// @return array of saved values
    std::array< float, BufferSize > get_buffer() const { return buffer; }

    /// @brief Returns the index of the most recent element in the buffer
    /// @return size_t index of the newest element in the buffer
    size_t get_head_index() const { return head; }

    /// @brief Returns the value of the most recent element in the buffer
    /// @return size_t index of the newest element in the buffer
    float get_head_value() const { return buffer[head]; }

    float get_mean_value() const {
        const float sum = std::accumulate(buffer.begin(), buffer.end(), 0.0F);
        const float average = sum / static_cast< float >(BufferSize);
        return average;
    }

    size_t get_number_elt() const { return numbers_elt; };
    bool filled() const { return numbers_elt >= BufferSize; }
    void reset() {
        head = 0;
        numbers_elt = 0;
        buffer.fill(0.0F);
    }

  private:
    size_t head{0};
    size_t numbers_elt{0};
    std::array< float, BufferSize > buffer{0};
};

class PredictionAggregator {
  public:
    /// @brief default constructor
    PredictionAggregator() = default;

    /// @brief save new observation
    void update_observation(const positionsScores& model_scores, const positionsMeasures& keypoints_measures);

    positionsResult get_aggregated_prediction() const { return current_output; }
    positionsMeasures get_framewise_measures() const;
    positionsScores get_framewise_model_scores() const;
    void reset() {
        closetoairbagMLBuffer.reset();
        pelvisrotationMLBuffer.reset();
        torsodissimetryMLBuffer.reset();
        oneshoulderincontactMLBuffer.reset();
        pelvisdriftMLBuffer.reset();
        legsondashboardMLBuffer.reset();
        reachingbackMLBuffer.reset();
        lateralbodycontactMLBuffer.reset();
        dashboardDistanceBuffer.reset();
        legsOrientationBuffer.reset();
        shoulderOrientationBuffer.reset();
        current_output = {};
    }

  private:
    void aggregate_predictions();
    void aggregate_closetoairbag();
    void aggregate_pelvisrotation();
    void aggregate_torsodissimetry();
    void aggregate_oneshoulderincontact();
    void aggregate_pelvisdrift();
    void aggregate_legsondashboard();
    void aggregate_reachingback();
    void aggregate_lateralbodycontact();

  private:
    floatBuffer< poseDefaultBufferSize > closetoairbagMLBuffer{};
    floatBuffer< poseDefaultBufferSize > pelvisrotationMLBuffer{};
    floatBuffer< poseDefaultBufferSize > torsodissimetryMLBuffer{};
    floatBuffer< poseDefaultBufferSize > oneshoulderincontactMLBuffer{};
    floatBuffer< poseDefaultBufferSize > pelvisdriftMLBuffer{};
    floatBuffer< poseDefaultBufferSize > legsondashboardMLBuffer{};
    floatBuffer< poseDefaultBufferSize > reachingbackMLBuffer{};
    floatBuffer< poseDefaultBufferSize > lateralbodycontactMLBuffer{};

    floatBuffer< poseDefaultBufferSize > dashboardDistanceBuffer{};
    floatBuffer< poseDefaultBufferSize > legsOrientationBuffer{};
    floatBuffer< poseDefaultBufferSize > shoulderOrientationBuffer{};

    positionsResult current_output{};
};

} // namespace foms

#endif // FOMS_POSITIONS_HPP