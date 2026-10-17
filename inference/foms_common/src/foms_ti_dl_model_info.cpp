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

#include "foms_ti_model_info.hpp"
#include "foms_logStream.hpp"

namespace tidl::modelInfo {

void InfererConfig::dumpInfo() const {

    foms::g_logger.writeLog("InfererConfig::Model Path = " + modelFile, foms::errorLevel::INFO);
    foms::g_logger.writeLog("InfererConfig::Artifacts Path = " + artifactsPath, foms::errorLevel::INFO);
    foms::g_logger.writeLog("InfererConfig::Runtime API = " + rtType, foms::errorLevel::INFO);
    foms::g_logger.writeLog("InfererConfig::Device Type = " + devType, foms::errorLevel::INFO);
}

int32_t getInfererConfig(const YAML::Node& appConfig, const std::string& modelBasePath,
                         InfererConfig& infConfig) {
    const YAML::Node& node = appConfig["session"];
    int32_t status = 0;

    /* Validate the parsed yaml configuration and create the
    configuration for the inference object creation. */
    if (!node) {

        foms::g_logger.writeLog("Inference configuration parameters  missing.", foms::errorLevel::ERROR);
        status = -1;
        return status;
    }

    if (!node["model_path"]) {

        foms::g_logger.writeLog("Please specifiy a valid model path.", foms::errorLevel::ERROR);
        status = -1;
        return status;
    }

    if (!node["artifacts_folder"]) {

        foms::g_logger.writeLog("Artifacts directory path missing.", foms::errorLevel::ERROR);
        status = -1;
        return status;
    }
    if (!node["session_name"]) {

        foms::g_logger.writeLog("Please specifiy a valid run-time API type", foms::errorLevel::ERROR);
        status = -1;
        return status;
    }

    /*Initialize the inference configuration parameter object*/
    infConfig.rtType = node["session_name"].as< std::string >();

    if (infConfig.rtType == "tvmdlr") {
        infConfig.modelFile = modelBasePath + "/model";
    } else if ((infConfig.rtType == "tflitert") || (infConfig.rtType == "onnxrt")) {
        const std::string& modSrc = node["model_path"].as< std::string >();
        infConfig.modelFile = modelBasePath + "/" + modSrc;
    }

    const std::string& artSrc = node["artifacts_folder"].as< std::string >();
    infConfig.artifactsPath = modelBasePath + "/" + artSrc;

    if (node["device_type"]) {
        infConfig.devType = node["device_type"].as< std::string >();

    } else {
        infConfig.devType = "CPU";
    }

    if (node["device_id"]) {
        infConfig.devId = node["device_id"].as< int32_t >();

    } else {
        infConfig.devId = 0;
    }

    if (node["use_keypoints"]) {
        infConfig.use_keypoints = node["use_keypoints"].as< bool >();

    } else {
        infConfig.use_keypoints = true;
    }
    return status;
}

void ModelInfo::dumpInfo() const {

    foms::g_logger.writeLog("ModelInfo::modelPath  =  " + m_modelPath, foms::errorLevel::INFO);
}

ModelInfo::ModelInfo(std::string_view modelPath) : m_modelPath{modelPath} {}

int32_t ModelInfo::initialize() {
    YAML::Node yaml;
    int32_t status = 0;

    const std::string& configFile = m_modelPath + "/param.yaml";

    /* Check if the specified configuration file exists */
    std::ifstream file(configFile);
    if (!file.is_open()) {
        foms::g_logger.writeLog("Labels file : " + configFile + " not found\n", foms::errorLevel::ERROR);
        status = -1;
    }
    file.close();

    if (status == 0) {
        yaml = YAML::LoadFile(configFile);

        /* Populate m_infConfig from yaml */
        status = getInfererConfig(yaml, m_modelPath, m_infConfig);
    }

    return status;
}

} // namespace tidl::modelInfo
