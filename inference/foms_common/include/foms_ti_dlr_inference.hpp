#ifndef FOMS_TIDLRINFERENCE_HPP
#define FOMS_TIDLRINFERENCE_HPP

#include "foms_inference.hpp"
#include "foms_ti_model_info.hpp"
#include "itidl_rt.h"
#include <dlpack/dlpack.h>
#include <dlr.h>

/**
 * @file
 * @brief Defines  class for model inference with TexasInstruments dlr runtime.
 */

namespace foms {

/// @brief class for neural network model inference with ti dlr runtime

class TIDLRInferenceObject : public InferenceObject {

  public:
    /// @brief default constructor
    TIDLRInferenceObject();

    /// @brief Constructor with a model file
    /// @param modelfile path of the model file
    explicit TIDLRInferenceObject(std::string_view modelfile);

    /// @brief destructor
    ~TIDLRInferenceObject() override;

    /// @brief Deleted copy constructor
    TIDLRInferenceObject(const TIDLRInferenceObject&) = delete;

    /// @brief Deleted copy assignment operator
    TIDLRInferenceObject& operator=(const TIDLRInferenceObject&) = delete;

    /// @brief performs the model inference with uint8 data as input
    /// @param data: the image data
    /// @return true if the inference is done, false else
    bool inferenceOnImage(uint8_t* data) override;

    /// @brief get the size of the output with the given index
    /// @param index : index of the output
    /// @return the size of the output
    size_t getOutputSize(int index) const override;

    /// @brief get the output at a given index
    /// @param index : index of the output
    /// @return  a float pointer to the output data
    float* getOutput(int index) override;

    /// @brief get the input size
    /// @return a vector containing the input size dimensions
    std::vector< int64_t > getInputSize() const override { return m_input_shape; };

  private:
    /// @brief Handle to the DLR model
    DLRModelHandle m_model = nullptr;

    /// @brief Information about the TIDL model
    tidl::modelInfo::ModelInfo m_modelInfo;

    /// @brief Name of the input tensor
    const char* m_input_name{nullptr};

    /// @brief Size of the input tensor
    int64_t m_input_size = 0;

    /// @brief Dimension of the input tensor
    int m_input_dim = 0;

    /// @brief Number of outputs from the model
    int m_num_outputs = 0;

    /// @brief Shape of the input tensor
    std::vector< int64_t > m_input_shape;

    /// @brief Sizes of the output tensors
    std::vector< int64_t > m_output_sizes;

    /// @brief Output data from the model
    std::vector< std::vector< float > > m_outputs;

    /// @brief Image data for inference
    uint8_t* m_image_data = nullptr;

    /// @brief DLTensor structure for input data
    DLTensor m_dltensor = {};
};

} // namespace foms

#endif // FOMS_TIDLRINFERENCE_HPP