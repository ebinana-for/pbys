/*

modified by Probayes for Forvia using original file from Texas Intstrument
see licence below

 */

/*
Copyright (c) 2020 – 2021 Texas Instruments Incorporated

All rights reserved not granted herein.

Limited License.

Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
license under copyrights and patents it now or hereafter owns or controls to
make, have made, use, import, offer to sell and sell ("Utilize") this software
subject to the terms herein.  With respect to the foregoing patent license,
such license is granted  solely to the extent that any such patent is necessary
to Utilize the software alone.  The patent license shall not apply to any
combinations which include this software, other than combinations with devices
manufactured by or for TI (“TI Devices”).  No hardware patent is licensed
hereunder.

Redistributions must preserve existing copyright notices and reproduce this
license (including the above copyright notice and the disclaimer and
(if applicable) source code license limitations below) in the documentation
and/or other materials provided with the distribution

Redistribution and use in binary form, without modification, are permitted
provided that the following conditions are met:

*	No reverse engineering, decompilation, or disassembly of this software is
    permitted with respect to any software provided in binary form.

*	any redistribution and use are licensed by TI for use only with TI Devices.

*	Nothing shall obligate TI to provide you with source code for the software
    licensed and provided to you in object code.

If software source code is provided to you, modification and redistribution of
the source code are permitted provided that the following conditions are met:

*	any redistribution and use of the source code, including any resulting
    derivative works, are licensed by TI for use only with TI Devices.

*	any redistribution and use of any object code compiled from the source code
    and any resulting derivative works, are licensed by TI for use only with TI
    Devices.

Neither the name of Texas Instruments Incorporated nor the names of its
suppliers may be used to endorse or promote products derived from this software
without specific prior written permission.

DISCLAIMER.

THIS SOFTWARE IS PROVIDED BY TI AND TI’S LICENSORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL TI AND TI’S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#ifndef _UTILS_MODEL_INFO_H_
#define _UTILS_MODEL_INFO_H_

/* Standard headers. */
#include <map>
#include <string>
#include <string_view>
#include <set>
#include <fstream>
#include <experimental/filesystem>
#include <iostream>

/* Third-party headers. */

#include <yaml-cpp/yaml.h>
#include "foms_logStream.hpp"

namespace tidl::modelInfo {

/**
 * @brief Configuration for the DL inferer.
 *
 */

struct InfererConfig {
    /** @brief Path to the model directory or a file.
     *  - file for TFLITE and ONNX
     *  - directory for DLR
     **/
    std::string modelFile{};

    /** @brief  Path to the directory containing the model artifacts. This is only
     *  valid for TFLITE models and is not looked at for the other ones.
     */
    std::string artifactsPath{};

    /** @brief  Type of the runtime API to invoke. The valid values are:
     * - DL_INFER_RTTYPE_DLR
     * - DL_INFER_RTTYPE_TFLITE
     * - DL_INFER_RTTYPE_ONNX
     */
    std::string rtType{};

    /** @brief  Type of the device. This field is specific to the DLR API and is
     * is not looked at for the other ones. Please refer to the DLR API
     * specification for valid values this field can take.
     */
    std::string devType{};

    /** @brief Id of the device. This field is specific to the DLR API and is
     * is not looked at for the other ones. Please refer to the DLR API
     * specification for valid values this field can take.
     */
    int32_t devId{-1};

    /** @brief use keypoints for inference.
     * This is a boolean flag that indicates whether the inference engine
     * should use keypoints for inference or not. If set to true, the inference
     * engine will use keypoints for inference. If set to false, the inference
     * engine will not use keypoints for inference.
     * Default value is true.
     */
    bool use_keypoints{true};

    /**
     * @brief  Helper function to dump the configuration information.
     */
    void dumpInfo() const;
};

/**
 * @brief  Enumeration for the different data types used for identifying
 *        the data types at the interface.
 *
 *
 */
enum class DlInferType {
    /**  Invalid type. */
    DlInferType_Invalid = 0,

    /** Data type signed 8 bit integer. */
    DlInferType_Int8 = 2,

    /** Data type unsigned 8 bit integer. */
    DlInferType_UInt8 = 3,

    /** Data type signed 16 bit integer. */
    DlInferType_Int16 = 4,

    /** Data type unsigned 16 bit integer. */
    DlInferType_UInt16 = 5,

    /** Data type signed 32 bit integer. */
    DlInferType_Int32 = 6,

    /** Data type unsigned 32 bit integer. */
    DlInferType_UInt32 = 7,

    /** Data type signed 64 bit integer. */
    DlInferType_Int64 = 8,

    /** Data type 16 bit floating point. */
    DlInferType_Float16 = 9,

    /** Data type 32 bit floating point. */
    DlInferType_Float32 = 10,
};

/**
 * @brief Class for holding DL inference model parameters for setting up
 *        inference pipeline flow.
 *
 * \ingroup group_edgeai_demo_config
 */
class ModelInfo {
  public:
    /** Default constructor. Use the compiler generated default one. */
    ModelInfo() = default;

    /** Constructor.
     *
     * @param m_modelPath model path.
     */
    explicit ModelInfo(std::string_view m_modelPath);

    /** Destructor. */
    ~ModelInfo() = default;

    /** Initializes the object. The following is done during
     * initialization:
     * - Parse the model specific param.yaml file and extract the
     *   model parameters.
     * - Instantiate an inference object
     * - Create the pre-processing
     */
    int32_t initialize();

    /**
     * Helper function to dump the configuration information.
     *
     *
     */
    void dumpInfo() const;

  public:
    /* inferer configuration.*/
    InfererConfig m_infConfig;

    /** Path to the model. */
    std::string m_modelPath;
};

/**
 * @brief  Function to parse the YAML configuration file and extract the
 *         inference configuration parameters.
 *
 * @param appConfig The YAML node containing the application configuration.
 * @param modelBasePath The base path for the model.
 * @param infConfig The InfererConfig object to populate with the parsed values.
 *
 * @return 0 on success, negative value on failure.
 */

int32_t getInfererConfig(const YAML::Node& appConfig, const std::string& modelBasePath,
                         InfererConfig& infConfig);

} // namespace tidl::modelInfo

#endif //_UTILS_MODEL_INFO_H_
