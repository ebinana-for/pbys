#ifndef FOMS_INFERENCE_HPP
#define FOMS_INFERENCE_HPP

#include <vector>
#include <string>
#include <iostream>
#include <sys/time.h>
#include <stdint.h>
/**
 * @file
 * @brief Defines abstract class for model inference.
 */

namespace foms {

/**
 * @brief A base class for neural network model inference.
 *
 * This class provides an interface for loading a model, performing inference on image data,
 * and retrieving inference results. It includes methods for checking if the model is loaded,
 * performing inference, and getting the size and data of the output.
 */
class InferenceObject {

  public:
    /// @brief Default constructor
    InferenceObject() = default;

    /**
     * @brief Constructor with a model file
     * @param modelfile Path of the model file
     */
    explicit InferenceObject(std::string_view modelfile);

    /// @brief Destructor
    virtual ~InferenceObject() = default;

    /**
     * @brief Checks if the model is loaded
     * @return true if the model is loaded, false otherwise
     */
    bool is_loaded() const { return m_model_loaded; }

    /**
     * @brief Performs the model inference with uint8 data as input
     * @param data The image data
     * @return true if the inference is done, false otherwise
     */
    virtual bool inferenceOnImage(uint8_t* data) = 0;

    /**
     * @brief Gets the size of the output with the given index
     * @param index Index of the output
     * @return The size of the output
     */
    virtual size_t getOutputSize(int index) const = 0;

    /**
     * @brief Gets the output at a given index
     * @param index Index of the output
     * @return A float pointer to the output data
     */
    virtual float* getOutput(int index) = 0;

    /**
     * @brief Pure virtual function to get the input size.
     * @return A vector containing the dimensions of the input.
     */
    virtual std::vector< int64_t > getInputSize() const = 0;

    /**
     * @brief Retrieves the model file path.
     * @return A string representing the path to the model file.
     */
    std::string getModelfile() const { return m_modelfile; }

  protected:
    /**
     * @brief Sets the load state of the model.
     *
     * This function updates the internal state to indicate whether the model
     * has been loaded or not.
     *
     * @param model_loaded A boolean value indicating the load state of the model.
     *                      - true: The model is loaded.
     *                      - false: The model is not loaded.
     */
    void setLoadState(bool model_loaded) { m_model_loaded = model_loaded; };

  private:
    /// @brief Model file
    std::string m_modelfile;

    /// @brief State of the model (loaded/not loaded)
    bool m_model_loaded = false;
};

/**
 * @brief Function for computing the max element
 * @param values Values of the tensor output
 * @param size Size of the tensor output
 * @return The index and value of the maximum element of the tensor
 */
std::pair< size_t, float > get_max_element(const float* values, size_t size);

/**
 * @brief Gets the current time in microseconds
 * @return The current time in microseconds
 */
double getexetime();

} // namespace foms

#endif // FOMS_INFERENCE_HPP