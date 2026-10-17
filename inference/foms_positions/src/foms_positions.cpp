#include "foms_positions.hpp"
#include "foms_logStream.hpp"
#include <cstring>
#include "foms_ti_dlr_inference.hpp"
#include "foms_utils.hpp"
#include <cmath>

foms::PositionClassification::PositionClassification(std::string_view modelfile,
                                                     const cameraParameters& cameraParams)
    : m_modelInfo{modelfile}, m_cameraParams{cameraParams} {

    if (m_modelInfo.initialize() < 0) {
        foms::g_logger.writeLog("PositionClassification not initialized, modelfile is not correct ",
                                foms::errorLevel::ERROR);
        return;
    }

    if (m_modelInfo.m_infConfig.rtType == "tvmdlr") {
        m_inferenceEngine.reset(new TIDLRInferenceObject(modelfile));

    } else {
        foms::g_logger.writeLog("PositionClassification not initialized, inference type not supported  ",
                                foms::errorLevel::ERROR);
        return;
    }
    const std::vector< int64_t > size_input = m_inferenceEngine->getInputSize();

    const auto maxSize = std::max(size_input[2], size_input[3]);
    m_data.resize(maxSize * maxSize);
    m_image_crop = cv::Mat(static_cast< int >(size_input[2]), static_cast< int >(size_input[3]), CV_8UC1);

    m_loaded = true;
    m_K.at< double >(0, 0) = cameraParams.focalLengthX;
    m_K.at< double >(1, 1) = cameraParams.focalLengthY;
    m_K.at< double >(0, 2) = cameraParams.principalPointX;
    m_K.at< double >(1, 2) = cameraParams.principalPointY;
    m_K.at< double >(2, 2) = 1.0;
    m_distCoeffs.at< double >(0) = cameraParams.distortionRadial2;
    m_distCoeffs.at< double >(1) = cameraParams.distortionRadial4;
    m_distCoeffs.at< double >(2) = cameraParams.distortionRadial6;

    g_logger.writeLog("PositionClassification initialized ", foms::errorLevel::INFO);
}

foms::positionsMeasures foms::PositionClassification::compute_measures_from_keypoints(
    const std::vector< foms::BodyKeyPoint >& bodyKeyPts) const {

    const float distance_to_board = compute_distance_from_board(bodyKeyPts);
    foms::positionsMeasures keypoint_metrics{};
    keypoint_metrics.dashboardDistance = distance_to_board;

    const foms::BodyKeyPoint& leftShoulder = bodyKeyPts[foms::LeftShoulderKeypointIndex];
    const foms::BodyKeyPoint& rightShoulder = bodyKeyPts[foms::RightShoulderKeypointIndex];

    const cv::Mat H = m_K * m_rotation * m_K.inv();
    const foms::BodyKeyPoint leftShoulder_trans = foms::transformBodyKeyPoint(leftShoulder, H);
    const foms::BodyKeyPoint rightShoulder_trans = foms::transformBodyKeyPoint(rightShoulder, H);

    if (leftShoulder_trans.valid && rightShoulder_trans.valid) {
        keypoint_metrics.shoulderOrientation =
            std::atan2(std::fabs(static_cast< float >(leftShoulder_trans.y - rightShoulder_trans.y)),
                       std::fabs(static_cast< float >(leftShoulder_trans.x - rightShoulder_trans.x)));
    } else {
        keypoint_metrics.shoulderOrientation = -1.0;
    }

    return keypoint_metrics;
}

float foms::PositionClassification::compute_distance_from_board(
    const std::vector< foms::BodyKeyPoint >& bodyKeyPts) const {

    const foms::BodyKeyPoint& nose = bodyKeyPts[foms::NoseKeypointIndex];

    if (!nose.valid) {
        return 0.0;
    }
    const double x = (nose.x - m_cameraParams.principalPointX) / m_cameraParams.focalLengthX;
    const double y = (nose.y - m_cameraParams.principalPointY) / m_cameraParams.focalLengthY;
    double z = 1.0;

    // Apply the camera rotation matrix
    const double xc = m_rotation.at< double >(0, 0) * x + m_rotation.at< double >(0, 1) * y +
                      m_rotation.at< double >(0, 2) * z;
    const double yc = m_rotation.at< double >(1, 0) * x + m_rotation.at< double >(1, 1) * y +
                      m_rotation.at< double >(1, 2) * z;
    const double zc = m_rotation.at< double >(2, 0) * x + m_rotation.at< double >(2, 1) * y +
                      m_rotation.at< double >(2, 2) * z;

    const double distance_lateral_left = m_cameraParams.distance_lateral_cameracenter_siegecenter_left;
    const double distance_lateral_right = m_cameraParams.distance_lateral_cameracenter_siegecenter_right;

    const double distance_board_camera_long = m_cameraParams.distance_board_camera_long;

    const double norm = sqrt(xc * xc + yc * yc + zc * zc);

    z /= norm;
    const double distance_lateral = (x > 0) ? distance_lateral_left : distance_lateral_right;

    // simplification
    // plane equation : x = +- distance_lateral_camera_center_siege

    const double factor = distance_lateral / std::fabs(x);

    // x *= factor; //  Clang tidy error. Is the logic correct?
    // y *= factor; // Clang tidy error
    z *= factor;

    const auto distance_to_board = static_cast< float >(fabs(z - distance_board_camera_long));

    return distance_to_board;
}

foms::positionsScores foms::PositionClassification::classify(uint8_t* data) const {

    foms::positionsScores classification_scores{};

    if (!is_loaded()) {
        g_logger.writeLog("PositionClassification not initialized, inference is not possible ",
                          foms::errorLevel::ERROR);
        return classification_scores;
    }
    if (!m_inferenceEngine->inferenceOnImage(data)) {
        g_logger.writeLog("Error in inference ", foms::errorLevel::ERROR);
        return classification_scores;
    }

    const float* result = m_inferenceEngine->getOutput(0);

    classification_scores.fillfromOutput(result);

    return classification_scores;
}
foms::positionsScores
foms::PositionClassification::classify(uint8_t* data, int width, int height, int nchannels,
                                       const foms::CropRegion& crop,
                                       const std::vector< foms::BodyKeyPoint >& bodyKeyPts) {

    foms::positionsScores classification_scores{};
    if (nchannels != 1) {
        g_logger.writeLog("invalid number of channels, inference is not possible ", foms::errorLevel::ERROR);
        return classification_scores;
    }
    const cv::Mat image(height, width, CV_8UC1, data);
    return classify(image, crop, bodyKeyPts);
}
foms::positionsScores
foms::PositionClassification::classify(const cv::Mat& image, const foms::CropRegion& crop,
                                       const std::vector< foms::BodyKeyPoint >& bodyKeyPts) {

    foms::positionsScores classification_scores{};
    if (!is_loaded()) {
        g_logger.writeLog("PositionClassification not initialized, inference is not possible ",
                          foms::errorLevel::ERROR);
        return classification_scores;
    }
    if (image.type() != CV_8UC1) {
        g_logger.writeLog("Unsupported image type ", foms::errorLevel::ERROR);
        return classification_scores;
    }
    if (!crop.isvalid(image.cols, image.rows)) {
        g_logger.writeLog("Crop is invalid ", foms::errorLevel::ERROR);
        return classification_scores;
    }
    if (bodyKeyPts.size() != foms::NumberOfBodyKeypoints) {
        g_logger.writeLog("Number of Body Keypoints is different from expected ", foms::errorLevel::ERROR);
        return classification_scores;
    }

    // Add a margin to the crop region
    const auto delta_width = static_cast< int >(0.1F * static_cast< float >(crop.xmax - crop.xmin));
    const auto xmin = std::max(0, crop.xmin - delta_width);
    const auto xmax = std::min(image.cols - 1, crop.xmax + delta_width);

    const cv::Rect ROI(xmin, crop.ymin, xmax - xmin, crop.ymax - crop.ymin);

    const std::vector< int64_t > inputSize = m_inferenceEngine->getInputSize();

    const int64_t in_cols = inputSize[3];
    const int64_t in_rows = inputSize[2];

    if (ROI.width != in_cols || ROI.height != in_rows) {

        const float scalex = static_cast< float >(in_cols) / static_cast< float >(ROI.width);
        const float scaley = static_cast< float >(in_rows) / static_cast< float >(ROI.height);
        const float scale = std::min(scalex, scaley);

        const auto dst_rows = static_cast< int >(std::lround(static_cast< float >(ROI.height) * scale));
        const auto dst_cols = static_cast< int >(std::lround(static_cast< float >(ROI.width) * scale));

        auto resizedImage = cv::Mat(dst_rows, dst_cols, CV_8UC1, m_data.data());
        cv::resize(image(ROI), resizedImage, cv::Size(dst_cols, dst_rows));

        const auto top = static_cast< int >((in_rows - resizedImage.rows) / 2);
        const auto bottom = static_cast< int >(in_rows - top - resizedImage.rows);
        const auto right = static_cast< int >((in_cols - resizedImage.cols) / 2);
        const auto left = static_cast< int >(in_cols - right - resizedImage.cols);

        cv::copyMakeBorder(resizedImage, m_image_crop, top, bottom, left, right, cv::BORDER_CONSTANT,
                           cv::Scalar(0));

        if (m_modelInfo.m_infConfig.use_keypoints) {
            // Draw keypoints on the resized image
            drawBodyKeypointsonImage(m_image_crop, bodyKeyPts, xmin, crop.ymin, scale, left, bottom);
        }
    } else {
        image(ROI).copyTo(m_image_crop);
        if (m_modelInfo.m_infConfig.use_keypoints) {
            drawBodyKeypointsonImage(m_image_crop, bodyKeyPts, xmin, crop.ymin);
        }
    }

    return classify(m_image_crop.data);
}

void foms::PositionClassification::drawBodyKeypointsonImage(
    cv::Mat& image, const std::vector< foms::BodyKeyPoint >& bodyKeyPts, int xoffset, int yoffset,
    float scale, int left, int top) const {

    cv::Scalar color{};

    if (bodyKeyPts.empty() || bodyKeyPts.size() != foms::NumberOfBodyKeypoints) {
        return;
    }

    if (image.type() == CV_8UC3) {
        color = cv::Scalar(255, 255, 255);
    } else {
        color = cv::Scalar(255);
    }
    const int lineSize = 3;
    for (const auto& index_values : foms::keyPointsLines) {

        const foms::BodyKeyPoint& p1 = bodyKeyPts[index_values[0]];
        const foms::BodyKeyPoint& p2 = bodyKeyPts[index_values[1]];

        if (p1.valid && p2.valid) {
            const auto p1x = static_cast< int >(
                std::lround(static_cast< float >(p1.x - xoffset) * scale + static_cast< float >(left)));
            const auto p1y = static_cast< int >(
                std::lround(static_cast< float >(p1.y - yoffset) * scale + static_cast< float >(top)));
            const auto p2x = static_cast< int >(
                std::lround(static_cast< float >(p2.x - xoffset) * scale + static_cast< float >(left)));
            const auto p2y = static_cast< int >(
                std::lround(static_cast< float >(p2.y - yoffset) * scale + static_cast< float >(top)));

            if (p1x < 0 || p1x >= image.cols || p1y < 0 || p1y >= image.rows || p2x < 0 ||
                p2x >= image.cols || p2y < 0 || p2y >= image.rows) {
                continue;
            }

            cv::line(image, cv::Point(p1x, p1y), cv::Point(p2x, p2y), color, lineSize);
        }
    }
};

foms::positionsResult
foms::PositionClassification::estimate(const cv::Mat& image, const foms::CropRegion& crop,
                                       const std::vector< foms::BodyKeyPoint >& bodyKeyPts) {
    const auto scores = classify(image, crop, bodyKeyPts);
    const auto keypoint_measures = compute_measures_from_keypoints(bodyKeyPts);
    return computeResultsFromScores(scores, keypoint_measures);
}

foms::positionsResult
foms::PositionClassification::computeResultsFromScores(const positionsScores& scores,
                                                       const positionsMeasures& measures) const {

    foms::positionsResult results{};
    results.valid = true;
    results.closetoairbag = scores.closetoairbag > poseDefaultModelThreshold;
    results.pelvisrotation = scores.pelvisrotation > poseDefaultModelThreshold;
    results.torsodissimetry = scores.torsodissimetry > poseDefaultModelThreshold;
    results.pelvisdrift = scores.pelvisdrift > poseDefaultModelThreshold;
    results.oneshoulderincontact = scores.oneshoulderincontact > poseDefaultModelThreshold;
    results.legsondashboard = scores.legsondashboard > poseDefaultModelThreshold;
    results.reachingback = scores.reachingback > poseDefaultModelThreshold;
    results.lateralbodycontact = scores.lateralbodycontact > poseDefaultModelThreshold;

    results.scores = scores;
    results.measures = measures;

    return results;
}

void foms::PositionClassification::set_camera_rotation(const cv::Mat& rotation) {
    if (rotation.rows != 3 || rotation.cols != 3) {

        g_logger.writeLog("Invalid rotation matrix size. Expected 3x3.", foms::errorLevel::ERROR);
        return;
    }
    m_rotation = rotation.clone();
}

void foms::positionsScores::fillfromOutput(const float* inferenceoutput) {
    valid = true;

    closetoairbag = sigmoid(inferenceoutput[static_cast< int >(foms::IndexClassPosition::CLOSETOAIRBAG)]);
    pelvisrotation = sigmoid(inferenceoutput[static_cast< int >(foms::IndexClassPosition::PELVISROTATION)]);
    torsodissimetry = sigmoid(inferenceoutput[static_cast< int >(foms::IndexClassPosition::TORSODISSIMETRY)]);
    oneshoulderincontact =
        sigmoid(inferenceoutput[static_cast< int >(foms::IndexClassPosition::ONESHOULDERINCONTACT)]);
    pelvisdrift = sigmoid(inferenceoutput[static_cast< int >(foms::IndexClassPosition::PELVISDRIFT)]);
    legsondashboard = sigmoid(inferenceoutput[static_cast< int >(foms::IndexClassPosition::LEGSONDASHBOARD)]);
    reachingback = sigmoid(inferenceoutput[static_cast< int >(foms::IndexClassPosition::REACHINGBACK)]);
    lateralbodycontact =
        sigmoid(inferenceoutput[static_cast< int >(foms::IndexClassPosition::LATERALBODYCONTACT)]);
}
