#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_log.hpp>
#include <boost/test/data/test_case.hpp>
#include <cstdint>
#include <vector>
#include <set>

#include "foms_logStream.hpp"
#include "foms_ti_dlr_inference.hpp"
#include "foms_utils.hpp"
#include "foms_csv_reader.hpp"
#include "foms_ti_model_info.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// // =============================================================================================
BOOST_AUTO_TEST_SUITE(_main_ti_model_info__class);
// // =============================================================================================

BOOST_AUTO_TEST_CASE(_ti_model_info_inferconfig) {
    const std::string modelPath = "../../../model-artifacts/model_age_gender";
    YAML::Node yaml;

    const std::string& configFile = modelPath + "/param.yaml";
    yaml = YAML::LoadFile(configFile);

    BOOST_CHECK(yaml.IsDefined() == true);

    tidl::modelInfo::InfererConfig infConfig;
    /* Populate m_infConfig from yaml */
    int32_t status = getInfererConfig(yaml, modelPath, infConfig);
    BOOST_CHECK(status == 0);
    yaml = YAML::LoadFile(configFile);
    yaml.remove("session");
    status = getInfererConfig(yaml, modelPath, infConfig);
    BOOST_CHECK(status != 0);

    yaml = YAML::LoadFile(configFile);
    yaml["session"].remove("model_path");
    status = getInfererConfig(yaml, modelPath, infConfig);
    BOOST_CHECK(status != 0);

    yaml = YAML::LoadFile(configFile);
    yaml["session"].remove("artifacts_folder");
    status = getInfererConfig(yaml, modelPath, infConfig);
    infConfig.dumpInfo();
    BOOST_CHECK(status != 0);

    yaml = YAML::LoadFile(configFile);
    yaml["session"].remove("session_name");
    status = getInfererConfig(yaml, modelPath, infConfig);
    BOOST_CHECK(status != 0);

    yaml = YAML::LoadFile(configFile);
    yaml["session"]["session_name"] = "tflitert";
    status = getInfererConfig(yaml, modelPath, infConfig);

    BOOST_CHECK(infConfig.modelFile ==
                "../../../model-artifacts/model_age_gender/model/model_age_gender.onnx");
    BOOST_CHECK(status == 0);

    yaml = YAML::LoadFile(configFile);
    yaml["session"].remove("device_type");
    status = getInfererConfig(yaml, modelPath, infConfig);
    BOOST_CHECK(infConfig.devType == "CPU");
    BOOST_CHECK(status == 0);

    yaml = YAML::LoadFile(configFile);
    yaml["session"]["device_type"] = "CPU";
    status = getInfererConfig(yaml, modelPath, infConfig);
    BOOST_CHECK(infConfig.devType == "CPU");
    BOOST_CHECK(status == 0);

    yaml = YAML::LoadFile(configFile);
    yaml["session"]["device_id"] = "1";
    status = getInfererConfig(yaml, modelPath, infConfig);
    BOOST_CHECK(infConfig.devId == 1);
    BOOST_CHECK(status == 0);
}
BOOST_AUTO_TEST_CASE(_ti_model_info_object) {
    const std::string modelPath = "../../../model-artifacts/model_age_gender";
    tidl::modelInfo::ModelInfo modelInfo{modelPath};
    BOOST_CHECK(modelInfo.m_modelPath == modelPath);
    const int32_t status = modelInfo.initialize();
    BOOST_CHECK(status == 0);
    modelInfo.dumpInfo();
}
BOOST_AUTO_TEST_SUITE_END();

// // =============================================================================================
BOOST_AUTO_TEST_SUITE(_main_ti_dlr_class);
// // =============================================================================================

BOOST_AUTO_TEST_CASE(_empty_ti_dlr_object) {
    foms::TIDLRInferenceObject object{};
    BOOST_CHECK(object.is_loaded() == false);
    uint8_t* data = nullptr;
    const bool ret = object.inferenceOnImage(data);
    BOOST_CHECK(ret == false);
}
BOOST_AUTO_TEST_CASE(_falsepath_ti_dlr_object) {
    const foms::TIDLRInferenceObject object{"test.xml"};
    BOOST_CHECK(object.is_loaded() == false);
}

BOOST_AUTO_TEST_CASE(_empty_ti_dlr_object_size_0) {
    const foms::TIDLRInferenceObject object{};
    BOOST_CHECK(object.getOutputSize(0) == 0);
}

BOOST_AUTO_TEST_CASE(_empty_ti_dlr_object_nullptr) {
    foms::TIDLRInferenceObject object{};
    BOOST_CHECK(object.getOutput(0) == nullptr);
}

BOOST_AUTO_TEST_CASE(_ti_dlr_object_model_loaded) {
    const foms::TIDLRInferenceObject object{"../../../model-artifacts/model_age_gender"};
    BOOST_CHECK(object.is_loaded());
}

BOOST_AUTO_TEST_CASE(ti_dlr_object_size) {
    const foms::TIDLRInferenceObject object{"../../../model-artifacts/model_age_gender"};
    BOOST_CHECK(object.is_loaded() && object.getOutputSize(0) == 3 && object.getOutputSize(1) == 0);
}

BOOST_AUTO_TEST_CASE(ti_dlr_object_ptr) {
    foms::TIDLRInferenceObject object{"../../../model-artifacts/model_age_gender"};
    BOOST_CHECK(object.is_loaded());
    BOOST_CHECK(object.getOutput(0) != nullptr && object.getOutput(1) == nullptr);
}

BOOST_AUTO_TEST_CASE(ti_dlr_object_inference_null_ptr) {
    foms::TIDLRInferenceObject object{"../../../model-artifacts/model_age_gender"};
    BOOST_CHECK(object.is_loaded());
    uint8_t* data = nullptr;
    const bool ret = object.inferenceOnImage(data);
    BOOST_CHECK(ret == false);
}

BOOST_AUTO_TEST_CASE(ti_dlr_object_inference) {
    foms::TIDLRInferenceObject object{"../../../model-artifacts/model_age_gender"};
    BOOST_CHECK(object.is_loaded());
    const std::string image_path = "../../../data/images/0.png";
    cv::Mat image{cv::imread(image_path)};
    BOOST_CHECK(image.rows != 0);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::resize(image, image, cv::Size(192, 224));

    const bool ret = object.inferenceOnImage(image.data);
    BOOST_CHECK(ret == true);
    const size_t output_size = object.getOutputSize(0);
    BOOST_CHECK(output_size == 3);
    const float* result = object.getOutput(0);
    BOOST_CHECK(result != nullptr);

    const std::pair< size_t, float > max_elt = foms::get_max_element(result, 2);
    BOOST_CHECK(max_elt.first == 0);
}

BOOST_AUTO_TEST_SUITE_END();

// =============================================================================================
BOOST_AUTO_TEST_SUITE(_csvreader_class);
// =============================================================================================

BOOST_AUTO_TEST_CASE(_csvreader_create) {
    const foms::CsvReader reader("../../../data/example.csv", ',');
    BOOST_CHECK(reader.is_loaded());
}

BOOST_AUTO_TEST_CASE(_csvreader_create_wrongfile) {
    const foms::CsvReader reader("../../../data/example_.csv", ',');
    BOOST_CHECK(!reader.is_loaded());
}
BOOST_AUTO_TEST_CASE(_csvreader_read_key_point) {
    const foms::CsvReader reader("../../../data/example.csv", ',');
    const std::string qualityName = foms::getBodykeypointQualityFieldName(0, 0);

    BOOST_CHECK(qualityName == "/trackingState/occupants/0/bodyKeyPoints/0/quality");
}

BOOST_AUTO_TEST_SUITE_END();

// =============================================================================================
BOOST_AUTO_TEST_SUITE(_utils_functions);
// =============================================================================================
BOOST_AUTO_TEST_CASE(_utils_read_key_point_all) {
    const foms::CsvReader reader("../../../data/example.csv", ',');
    BOOST_CHECK(reader.is_loaded());
    const foms::BodyKeyPoint keyptLeye1(reader, 0, 0, foms::LeftEyeKeypointIndex);
    const foms::BodyKeyPoint keyptReye1(reader, 0, 0, foms::RightEyeKeypointIndex);
    const foms::BodyKeyPoint keyptShin1(reader, 0, 0, foms::ShinKeypointsIndex);
    const foms::BodyKeyPoint keyptLear1(reader, 0, 0, foms::LeftEarKeypointIndex);
    const foms::BodyKeyPoint keyptRear1(reader, 0, 0, foms::RightEarKeypointIndex);

    BOOST_CHECK(keyptLeye1.valid && keyptReye1.valid && keyptShin1.valid && keyptLear1.valid &&
                keyptRear1.valid);
    foms::CropRegion facecrop{0, 224, 0, 224};
    const auto err = foms::computeFaceCropFromkeyPoints(keyptLeye1, keyptReye1, keyptShin1, keyptLear1,
                                                        keyptRear1, facecrop);
    BOOST_CHECK(facecrop.area() > 0);
    BOOST_CHECK(err == foms::FomsFaceErrorType::FomsFace_BADCROPSIZE);

    cv::Mat img = cv::Mat::zeros(800, 800, CV_8UC3);
    foms::drawEyesEarsShin(img, keyptLeye1, keyptReye1, keyptShin1, keyptLear1, keyptRear1,
                           cv::Scalar(0, 0, 255));

    std::cout << img.at< cv::Vec3b >(keyptLear1.y, keyptLear1.x) << std::endl;

    BOOST_CHECK(img.at< cv::Vec3b >(keyptLear1.y, keyptLear1.x) == cv::Vec3b(0, 0, 255));
}

BOOST_AUTO_TEST_CASE(_utils_read_bodybox) {
    const foms::CsvReader reader("../../../data/example.csv", ',');
    BOOST_CHECK(reader.is_loaded());
    BOOST_CHECK(reader.getNumberofLines() > 0); // Ensure there are lines to read
    BOOST_CHECK(reader.getValueByIndex(-1, 0).empty());
    BOOST_CHECK(reader.getValueByName("toto", 30).empty());
    const foms::BodyBBox bbox(reader, 0, 0);

    BOOST_CHECK(bbox.valid);
    BOOST_CHECK(bbox.getCrop().isvalid(5000, 5000));

    const foms::BodyBBox bbox_copy(bbox.x, bbox.y, bbox.width, bbox.height, bbox.valid, bbox.occupantid);
    BOOST_CHECK(bbox_copy.valid);
    BOOST_CHECK(bbox_copy.getCrop().isvalid(5000, 5000));
}

BOOST_AUTO_TEST_CASE(_utils_get_max_element) {
    std::vector< float > arr = {1.0F, 2.5F, 0.5F};
    auto max_elt = foms::get_max_element(arr.data(), 3);
    BOOST_CHECK(max_elt.first == 1);
    BOOST_CHECK_CLOSE(max_elt.second, 2.5F, 0.001);
}

BOOST_AUTO_TEST_CASE(_utils_cropregion_area) {
    const foms::CropRegion region{10, 20, 30, 40};
    BOOST_CHECK(region.area() == (20 - 10) * (40 - 30));
}
BOOST_AUTO_TEST_CASE(_utils_exetime) {
    const double time = foms::getexetime();
    BOOST_CHECK(time >= 0.0);
}

BOOST_AUTO_TEST_SUITE_END();
// =============================================================================================
BOOST_AUTO_TEST_SUITE(_logStream_class);
// =============================================================================================

BOOST_AUTO_TEST_CASE(_default_log_write_info) {
    foms::g_logger.writeLog("log", foms::errorLevel::INFO);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(_default_object_write_error) {
    foms::g_logger.writeLog("log", foms::errorLevel::ERROR);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(_changeverbosity) {
    foms::g_logger.set_verbosity(foms::LogVerbosity::NOLOG);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END();