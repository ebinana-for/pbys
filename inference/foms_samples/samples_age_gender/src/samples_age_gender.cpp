#include "foms_age_gender_estimator.hpp"
#include "foms_logStream.hpp"

#include <sstream>
// clang-format off

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// clang-format on
#include "foms_ti_dlr_inference.hpp"

int main() {

    foms::g_logger.set_verbosity(foms::LogVerbosity::ALL);
    foms::g_logger.writeLog("Gender and Age estimation sample", foms::errorLevel::INFO);
    ////Load the model////
    const std::string model_path = "../model-artifacts/model_age_gender";
    const std::string image_path = "../data/images/0.png";
    foms::AgeGenderEstimator object(model_path);

    if (!object.is_loaded()) {
        foms::g_logger.writeLog("Unable to load model , exit", foms::errorLevel::ERROR);
        return 0;
    }

    //////////Load the image///////////////
    const cv::Mat image = cv::imread(image_path);
    const foms::CropRegion crop(0, 224, 0, 224);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    if (image.empty()) {
        foms::g_logger.writeLog("Unable to load image , exit", foms::errorLevel::ERROR);
        return 0;
    }
    //////////Estimate the gender and age///////////////
    const auto estimation = object.estimate(image, crop);

    if (!estimation.valid) {
        foms::g_logger.writeLog("Unable to infer , exit", foms::errorLevel::ERROR);
        return 0;
    }
    std::stringstream ss{};
    ss << "Inference gender result : " << static_cast< int >(estimation.gender);
    foms::g_logger.writeLog(ss.str(), foms::errorLevel::INFO);

    std::stringstream ss_age{};
    ss_age << "Inference age result : " << static_cast< int >(estimation.age);
    foms::g_logger.writeLog(ss_age.str(), foms::errorLevel::INFO);
}