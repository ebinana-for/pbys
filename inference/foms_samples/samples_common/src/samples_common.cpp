#include "foms_inference.hpp"
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
    foms::g_logger.writeLog("Ceci est un sample de test d'inference avec ti_dlr", foms::errorLevel::INFO);

    const std::string model_path = "../model-artifacts/model_age_gender";
    const std::string image_path = "../data/images/0.png";
    foms::TIDLRInferenceObject object(model_path);

    if (!object.is_loaded()) {
        foms::g_logger.writeLog("Unable to load model , exit", foms::errorLevel::ERROR);
        return 0;
    }

    cv::Mat image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::resize(image, image, cv::Size(224, 224));

    bool ret = object.inferenceOnImage(image.data);
    if (!ret) {
        foms::g_logger.writeLog("Unable to infer , exit", foms::errorLevel::ERROR);
        return 0;
    }
    const size_t output_size = object.getOutputSize(0);
    std::cout << "size output " << output_size << "\n";
    const float* result = object.getOutput(0);
    if (result == nullptr) {
        foms::g_logger.writeLog("Unable to get output , exit", foms::errorLevel::ERROR);
        return 0;
    }
    const auto [gender, gender_conf] = foms::get_max_element(result, 2);

    std::stringstream ss{};
    ss << "Inference gender result, Max Val : " << gender << " " << gender_conf;
    foms::g_logger.writeLog(ss.str(), foms::errorLevel::INFO);

    foms::g_logger.writeLog("Benchmarking", foms::errorLevel::INFO);
    const int nIterations = 100;

    const double t1 = foms::getexetime();
    for (int i = 0; i < nIterations; i++) {
        ret = object.inferenceOnImage(image.data);
        if (!ret) {
            break;
        }
    }
    const double t2 = foms::getexetime();
    const double timeperIter((t2 - t1) / nIterations);
    std::stringstream ss2{};
    ss2 << "Inference time : " << timeperIter << " us";
    foms::g_logger.writeLog(ss2.str(), foms::errorLevel::INFO);
}