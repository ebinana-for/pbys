#include "foms_ti_dlr_inference.hpp"
#include "foms_logStream.hpp"
#include <cstring>

foms::TIDLRInferenceObject::TIDLRInferenceObject() : InferenceObject() {}

foms::TIDLRInferenceObject::TIDLRInferenceObject(std::string_view modelfile)
    : InferenceObject{modelfile}, m_modelInfo{modelfile} {

    if (m_modelInfo.initialize() < 0) {
        g_logger.writeLog("TIDLRInferenceObject not initialized : " + getModelfile(),
                          foms::errorLevel::ERROR);

        return;
    }
    if (CreateDLRModel(&m_model, m_modelInfo.m_infConfig.artifactsPath.c_str(), 1, 0) != 0) {
        g_logger.writeLog("Could not load model", foms::errorLevel::ERROR);
        return;
    }

    /* Query input name. */

    if (GetDLRInputName(&m_model, 0, &m_input_name) < 0) {
        const std::string error(DLRGetLastError());

        g_logger.writeLog("GetDLRInputName(0) failed :  " + error, foms::errorLevel::ERROR);
        return;
    }

    GetDLRInputSizeDim(&m_model, 0, &m_input_size, &m_input_dim);
    m_input_shape.resize(m_input_dim);

    GetDLRInputShape(&m_model, 0, m_input_shape.data());

    GetDLRNumOutputs(&m_model, &m_num_outputs);

    m_output_sizes.resize(m_num_outputs);
    m_outputs.resize(m_num_outputs);
    for (int i = 0; i < m_num_outputs; i++) {

        int output_dim = 0;
        GetDLROutputSizeDim(&m_model, i, &m_output_sizes[i], &output_dim);
        m_outputs[i].resize(m_output_sizes[i]);
    }

    m_image_data = static_cast< uint8_t* >(TIDLRT_allocSharedMem(
        128,
        static_cast< int32_t >(sizeof(uint8_t) * m_input_shape[1] * m_input_shape[2] * m_input_shape[3])));

    if (m_image_data == nullptr) {
        const std::string error = DLRGetLastError();
        g_logger.writeLog("Could not allocate space for image data " + error, foms::errorLevel::ERROR);
        return;
    }

    m_dltensor.device = {kDLCPU, 0};
    m_dltensor.ndim = 4;
    m_dltensor.shape = m_input_shape.data();
    m_dltensor.strides = nullptr;
    m_dltensor.byte_offset = 0;
    m_dltensor.dtype = {kDLUInt, static_cast< uint8_t >(8), 1};
    m_dltensor.data = m_image_data;

    if (SetDLRInputTensorZeroCopy(&m_model, m_input_name, &m_dltensor) != 0) {
        const std::string error = DLRGetLastError();
        g_logger.writeLog("Could not set input" + error, foms::errorLevel::ERROR);
        return;
    }
    setLoadState(true);
}
foms::TIDLRInferenceObject::~TIDLRInferenceObject() {
    if (m_image_data != nullptr) {
        TIDLRT_freeSharedMem(m_image_data);
    }
    if (m_model != nullptr && DeleteDLRModel(&m_model) != 0) {
        g_logger.writeLog("Could not delete model", foms::errorLevel::ERROR);
    }
}

bool foms::TIDLRInferenceObject::inferenceOnImage(uint8_t* data) {

    if (!is_loaded()) {
        g_logger.writeLog("Inference is not possible, model is not loaded ", foms::errorLevel::ERROR);
        return false;
    }
    if (data == nullptr) {
        g_logger.writeLog("Inference is not possible, data is nullptr ", foms::errorLevel::ERROR);
        return false;
    }

    const int64_t size = m_input_shape[2] * m_input_shape[3] * m_input_shape[1];

    std::memcpy(m_image_data, data, size);

    if (RunDLRModel(&m_model) != 0) {
        g_logger.writeLog("Could not run ", foms::errorLevel::ERROR);
    }

    return true;
}

size_t foms::TIDLRInferenceObject::getOutputSize(int index) const {

    if (!is_loaded()) {
        g_logger.writeLog("getOutputSize is not possible, model is not loaded ", foms::errorLevel::ERROR);
        return 0;
    }

    if (index >= m_num_outputs || index < 0) {
        g_logger.writeLog("Model has no output with specified index", foms::errorLevel::ERROR);
        return 0;
    }

    return static_cast< size_t >(m_output_sizes[index]);
}

float* foms::TIDLRInferenceObject::getOutput(int index) {

    if (!is_loaded()) {
        g_logger.writeLog("getOutput is not possible, model is not loaded ", foms::errorLevel::ERROR);
        return nullptr;
    }

    if (index >= m_num_outputs || index < 0) {
        g_logger.writeLog("Model has no output with specified index", foms::errorLevel::ERROR);
        return nullptr;
    }

    if (GetDLROutput(&m_model, index, m_outputs[index].data()) != 0) {
        g_logger.writeLog("Could not get output", foms::errorLevel::ERROR);
        return nullptr;
    }
    return m_outputs[index].data();
};
