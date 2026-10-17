#ifndef FOMS_AGEGENDER_HPP
#define FOMS_AGEGENDER_HPP

#include "foms_inference.hpp"
#include "foms_ti_model_info.hpp"
#include "foms_utils.hpp"
#include "itidl_rt.h"
#include <dlpack/dlpack.h>
#include <dlr.h>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
 * @file
 * @brief Defines  class for age and gender estimation.
 */

namespace foms {

/**
 * @brief A structure to hold age and gender estimation results.
 */
struct AgeGenderEstimation {
    uint8_t gender{0};        ///< The estimated gender. 0 represents male, and 1 represents female.
    uint8_t age{0};           ///< The estimated age.
    float score_age{0.0f};    ///< The confidence score for the age estimation.
    float score_gender{0.0f}; ///< The confidence score for the gender estimation.
    bool valid{false};        ///< A flag indicating whether the estimation is valid.
};

/// @brief Class for neural network age and model model inference with ti dlr runtime
/// The AgeGenderEstimator class provides functionality to estimate age and gender
/// from images using a neural network model.
class AgeGenderEstimator {

  public:
    /// @brief Deleted copy constructor to prevent copying.
    AgeGenderEstimator(const AgeGenderEstimator&) = delete;
    /// @brief Deleted copy assignment operator to prevent copying.
    AgeGenderEstimator& operator=(const AgeGenderEstimator&) = delete;

    /// @brief Default constructor.
    AgeGenderEstimator() = default;

    /// @brief Constructor with a model file.
    /// @param modelfile Path of the model file.
    explicit AgeGenderEstimator(std::string_view modelfile);

    /// @brief Destructor.
    ~AgeGenderEstimator();

    /// @brief Estimate age and gender from an image.
    /// @param image Input image.
    /// @param crop Region of the image to crop for estimation.
    /// @return AgeGenderEstimation containing the estimation results.
    AgeGenderEstimation estimate(const cv::Mat& image, const foms::CropRegion& crop);

    /// @brief Estimate age and gender from raw data.
    /// @param data Pointer to the raw image data.
    /// @return AgeGenderEstimation containing the estimation results.
    AgeGenderEstimation estimate(uint8_t* data);

    /// @brief Check if the model is loaded.
    /// @return True if the model is loaded, false otherwise.
    bool is_loaded() const { return m_loaded; }

    /// @brief Reset the estimator by clearing all previous estimations.
    void reset_estimator() { m_estimations.clear(); };

    /// @brief Get the mean estimation from all previous estimations.
    /// @return AgeGenderEstimation containing the mean estimation results.
    AgeGenderEstimation get_mean_estimation() const;

    /// @brief sets the camera rotation matrix
    /// @param rotation The rotation matrix to set
    /// @note The rotation matrix is expected to be a 3x3 matrix.
    void set_camera_rotation(const cv::Mat& rotation);

    /// @brief Get the camera rotation matrix.
    /// @return The camera rotation matrix as a cv::Mat.
    cv::Mat get_camera_rotation() const { return m_rotation; }

  private:
    tidl::modelInfo::ModelInfo m_modelInfo{};             ///< Model information.
    bool m_loaded = false;                                ///< Flag indicating if the model is loaded.
    std::shared_ptr< InferenceObject > m_inferenceEngine; ///< Inference engine for model inference.
    std::list< AgeGenderEstimation > m_estimations;       ///< List of previous estimations.

    cv::Mat m_image_crop; ///< Cropped image for internal use

    std::vector< uint8_t > m_data; ///< internal data process.

    cv::Mat m_rotation{cv::Mat::eye(3, 3, CV_64F)}; ///< Camera rotation matrix.
};

} // namespace foms

#endif // FOMS_AGEGENDER_HPP