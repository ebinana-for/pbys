#ifndef FOMS_POSITIONS_HPP
#define FOMS_POSITIONS_HPP

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
 * @brief Defines  class for position classification.
 */

namespace foms {

inline float sigmoid(const float x) { return 1.0F / (1.0F + expf(-x)); }

/// @brief Enum class representing different position indices for classification
enum class IndexClassPosition : uint8_t {
    CLOSETOAIRBAG = 0,        ///< Position close to the airbag
    LEGSONDASHBOARD = 1,      ///< Legs on dashboard position
    TORSODISSIMETRY = 2,      ///< Torso dissimetry position
    REACHINGBACK = 3,         ///< Reaching back position
    LATERALBODYCONTACT = 4,   ///< Lateral body contact position
    ONESHOULDERINCONTACT = 5, ///< One shoulder in contact position
    PELVISROTATION = 6,       ///< Pelvis rotation position
    PELVISDRIFT = 7,          ///< Pelvis drift position

};

/// @brief Struct to hold scores for different positions
struct positionsScores {
    float closetoairbag{0.0};        ///< Score for close to airbag position
    float pelvisrotation{0.0};       ///< Score for pelvis rotation position
    float torsodissimetry{0.0};      ///< Score for torso dissimetry position
    float oneshoulderincontact{0.0}; ///< Score for one shoulder in contact position
    float pelvisdrift{0.0};          ///< Score for pelvis drift in contact position
    float legsondashboard{0.0};      ///< Score for legs on dashboard position
    float reachingback{0.0};         ///< Score for reaching back position
    float lateralbodycontact{0.0};   ///< Score for lateral body contact position
    bool valid{false};               ///< Flag indicating if the scores are valid

    /// @brief Fills the scores from the output of the inference
    /// @param inferenceoutput Pointer to the inference output data
    void fillfromOutput(const float* inferenceoutput);
};

/// @brief Struct to hold measures
struct positionsMeasures {
    float dashboardDistance{100.0}; ///< distance measure for close to airbag position
    float legsOrientation{0.0};     ///< angle measure for pelvis rotation position
    float shoulderOrientation{0.0}; ///< angle measure for torso dissimetry position
};

/// @brief Struct to hold results for different positions
struct positionsResult {
    bool closetoairbag{false};        ///< Result for close to airbag position
    bool pelvisrotation{false};       ///< Result for pelvis rotation position
    bool torsodissimetry{false};      ///< Result for torso dissimetry position
    bool oneshoulderincontact{false}; ///< Result for one shoulder in contact position
    bool pelvisdrift{false};          ///< Result for pelvis drift position
    bool legsondashboard{false};      ///< Result for legs on dashboard position
    bool reachingback{false};         ///< Result for reaching back position
    bool lateralbodycontact{false};   ///< Result for lateral body contact position

    positionsScores scores;     ///< Scores for the positions classification
    positionsMeasures measures; ///< Measures for the positions classification

    bool valid{false}; ///< Flag indicating if the results are valid
};

/// @brief class for neural network model inference with ti dlr runtime
/// @class PositionClassification
/// @brief A class for classifying positions using a specified model file.
///
/// This class provides methods to classify positions based on image data and body key points.
class PositionClassification {

  public:
    /// @brief default constructor
    PositionClassification() = default;
    /// @brief Constructor with a model file
    /// @param modelfile path of the model file
    /// @param cameraParams Camera parameters for image processing
    explicit PositionClassification(std::string_view modelfile, const cameraParameters& cameraParams = {});
    /// @brief destructor
    ~PositionClassification() = default;

    /// @brief Deleted copy constructor
    PositionClassification(const PositionClassification&) = delete;
    /// @brief Deleted copy assignment operator
    PositionClassification& operator=(const PositionClassification&) = delete;

    // TODO can classify methods be private?

    /// @brief Classifies positions based on raw image data and body key points
    /// @param data Pointer to the raw image data
    /// @param width Width of the image
    /// @param height Height of the image
    /// @param nchannels Number of channels in the image
    /// @param crop Crop region of the image
    /// @param bodyKeyPts Vector of body key points
    /// @return positionsResult Struct containing the positions classification results
    positionsScores classify(uint8_t* data, int width, int height, int nchannels,
                             const foms::CropRegion& crop,
                             const std::vector< foms::BodyKeyPoint >& bodyKeyPts);

    /// @brief Classifies positions based on an OpenCV image and body key points
    /// @param image OpenCV image
    /// @param crop Crop region of the image
    /// @param bodyKeyPts Vector of body key points
    /// @return positionsResult Struct containing the positions classification results
    positionsScores classify(const cv::Mat& image, const foms::CropRegion& crop,
                             const std::vector< foms::BodyKeyPoint >& bodyKeyPts);

    /// @brief Classifies positions based on raw image data
    /// @param data Pointer to the raw image data
    /// @return positionsScores Struct containing the classification scores
    positionsScores classify(uint8_t* data) const;

    /// @brief Checks if the model is loaded
    /// @return True if the model is loaded, false otherwise
    bool is_loaded() const { return m_loaded; }

    positionsMeasures
    compute_measures_from_keypoints(const std::vector< foms::BodyKeyPoint >& bodyKeyPts) const;

    float compute_distance_from_board(const std::vector< foms::BodyKeyPoint >& bodyKeyPts) const;

    positionsResult estimate(const cv::Mat& image, const foms::CropRegion& crop,
                             const std::vector< foms::BodyKeyPoint >& bodyKeyPts);

    /// @brief sets the camera rotation matrix
    /// @param rotation The rotation matrix to set
    /// @note The rotation matrix is expected to be a 3x3 matrix.
    void set_camera_rotation(const cv::Mat& rotation);

    /// @brief Get the camera rotation matrix.
    /// @return The camera rotation matrix as a cv::Mat.
    cv::Mat get_camera_rotation() const { return m_rotation; }

  private:
    /**
     * @brief Computes the results from the given scores.
     *
     * This function takes a positionsScores object as input and computes the
     * corresponding results, returning them as a positionsResult object.
     *
     * @param scores The input scores used to compute the results.
     * @param scores The input measures associated to the psotions.
     * @return positionsResult The computed results based on the input scores.
     */
    positionsResult computeResultsFromScores(const positionsScores& scores,
                                             const positionsMeasures& measures) const;

    /// @brief Draws body key points on an image
    /// @param image OpenCV image
    /// @param bodyKeyPts Vector of body key points
    /// @param xoffset X offset for drawing
    /// @param yoffset Y offset for drawing
    void drawBodyKeypointsonImage(cv::Mat& image, const std::vector< foms::BodyKeyPoint >& bodyKeyPts,
                                  int xoffset, int yoffset, float scale = 1.0F, int left = 0,
                                  int top = 0) const;

    tidl::modelInfo::ModelInfo m_modelInfo{};             ///< Model information
    bool m_loaded = false;                                ///< Flag indicating if the model is loaded
    std::shared_ptr< InferenceObject > m_inferenceEngine; ///< Inference engine object
    cameraParameters m_cameraParams{};                    ///< Camera parameters for image processing

    cv::Mat m_K = cv::Mat(3, 3, CV_64F);          ///< Camera intrinsic matrix
    cv::Mat m_distCoeffs = cv::Mat(1, 3, CV_64F); ///< Camera distortion coefficients

    cv::Mat m_image_crop;                           ///< Cropped image for internal use
    std::vector< uint8_t > m_data;                  ///< internal data process.
    cv::Mat m_rotation{cv::Mat::eye(3, 3, CV_64F)}; ///< Camera rotation matrix.
};

} // namespace foms

#endif // FOMS_POSITIONS_HPP