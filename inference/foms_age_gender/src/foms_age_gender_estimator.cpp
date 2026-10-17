#include "foms_age_gender_estimator.hpp"
#include "foms_logStream.hpp"
#include <cstring>
#include "foms_ti_dlr_inference.hpp"
#include "foms_utils.hpp"
#include <cmath>

foms::AgeGenderEstimator::AgeGenderEstimator(std::string_view modelfile) : m_modelInfo{modelfile} {

    if (m_modelInfo.initialize() < 0) {
        foms::g_logger.writeLog("AgeGenderEstimator not initialized, modelfile is not correct ",
                                foms::errorLevel::ERROR);
        return;
    }

    if (m_modelInfo.m_infConfig.rtType == "tvmdlr") {
        m_inferenceEngine.reset(new TIDLRInferenceObject(modelfile));

    } else {
        foms::g_logger.writeLog("AgeGenderEstimator not initialized, inference type not supported  ",
                                foms::errorLevel::ERROR);
        return;
    }
    const std::vector< int64_t > size_input = m_inferenceEngine->getInputSize();

    const auto maxSize = std::max(size_input[2], size_input[3]);
    m_data.resize(maxSize * maxSize);
    m_image_crop = cv::Mat(static_cast< int >(size_input[2]), static_cast< int >(size_input[3]), CV_8UC1);

    m_loaded = true;

    g_logger.writeLog("AgeGenderEstimator initialized ", foms::errorLevel::INFO);
}
foms::AgeGenderEstimator::~AgeGenderEstimator() { m_loaded = false; }
foms::AgeGenderEstimation foms::AgeGenderEstimator::estimate(uint8_t* data) {

    foms::AgeGenderEstimation estimation{};

    if (!is_loaded()) {
        g_logger.writeLog("AgeGenderEstimator not initialized, inference is not possible ",
                          foms::errorLevel::ERROR);
        return estimation;
    }
    if (!m_inferenceEngine->inferenceOnImage(data)) {
        g_logger.writeLog("Error in inference ", foms::errorLevel::ERROR);
        return estimation;
    }

    const float* result = m_inferenceEngine->getOutput(0);

    const auto [gender, gender_conf] = foms::get_max_element(result, 2);

    const float gender_score = std::exp(result[gender]) / (std::exp(result[0]) + std::exp(result[1]));

    estimation.gender = static_cast< uint8_t >(gender);
    estimation.age = static_cast< uint8_t >(std::lround(result[2]));

    estimation.valid = true;
    estimation.score_age = 0.5F;
    estimation.score_gender = gender_score;
    m_estimations.push_back(estimation);
    return estimation;
}

foms::AgeGenderEstimation foms::AgeGenderEstimator::estimate(const cv::Mat& image,
                                                             const foms::CropRegion& crop) {

    foms::AgeGenderEstimation estimation{};

    if (!is_loaded()) {
        g_logger.writeLog("AgeGenderEstimator not initialized, inference is not possible ",
                          foms::errorLevel::ERROR);
        return estimation;
    }
    if (image.type() != CV_8UC1) {
        g_logger.writeLog("Unsupported image type ", foms::errorLevel::ERROR);
        return estimation;
    }
    if (!crop.isvalid(image.cols, image.rows)) {
        g_logger.writeLog("Crop is invalid ", foms::errorLevel::INFO);

        return estimation;
    }
    const cv::Rect ROI(crop.xmin, crop.ymin, crop.xmax - crop.xmin, crop.ymax - crop.ymin);

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

    } else {
        image(ROI).copyTo(m_image_crop);
    }

    return estimate(m_image_crop.data);
}

void foms::AgeGenderEstimator::set_camera_rotation(const cv::Mat& rotation) {
    if (rotation.rows != 3 || rotation.cols != 3) {

        g_logger.writeLog("Invalid rotation matrix size. Expected 3x3.", foms::errorLevel::ERROR);
        return;
    }
    m_rotation = rotation.clone();
}

foms::AgeGenderEstimation foms::AgeGenderEstimator::get_mean_estimation() const {
    foms::AgeGenderEstimation estimation{};

    if (m_estimations.empty()) {
        return estimation;
    }

    float mean_age = 0.0;
    int countF = 0;
    const auto count = static_cast< int >(m_estimations.size());

    for (const auto& est : m_estimations) {
        mean_age += static_cast< float >(est.age);
        countF += static_cast< int >(est.gender);
    }

    estimation.valid = true;
    mean_age /= static_cast< float >(count);
    estimation.age = static_cast< uint8_t >(std::lround(mean_age));
    estimation.gender = (countF > count / 2) ? 1 : 0;

    const float addedscore =
        std::min(1.0F, 1.0F * (static_cast< float >(count)) / maxNumberFrameAgeAndGender);
    estimation.score_age = 0.5F + 0.5F * (count > minNumberFrameAgeAndGender ? addedscore : 0.0F);
    estimation.score_gender = 0.5F + 0.5F * (count > minNumberFrameAgeAndGender ? addedscore : 0.0F);

    return estimation;
}
