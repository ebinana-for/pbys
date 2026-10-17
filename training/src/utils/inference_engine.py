"""Inference Engine for ONNX, MNN and TensorRT models
This module provides an abstract base class for inference engines that can handle ONNX, MNN, and TensorRT models.
It includes methods for loading models, preprocessing inputs, and postprocessing outputs.
"""

import logging
import os
from abc import abstractmethod
from enum import Enum

import numpy as np

log = logging.getLogger("Inference")
def errorstring(engine):
    return f"Cannot import {engine}. {engine} inference engine will not be tested."

try:
    import onnxruntime
    onnxruntime_loaded = True
except ModuleNotFoundError:
    onnxruntime_loaded = False
try:
    import MNN
    mnn_loaded = True
except ModuleNotFoundError:
    mnn_loaded = False
try:
    import tensorrt as trt
    trt_loaded = True
except ModuleNotFoundError:
    trt_loaded = False
try:
    import pycuda.driver as cuda
    pycuda_loaded = True
except ModuleNotFoundError:
    pycuda_loaded = False


class ModelType(Enum):
    """Used to convert from extension to the type of Model"""

    ONNX = ".onnx"
    MNN = ".mnn"
    TRT = ".plan"


class InferenceEngine:
    """Abstract base class for inference engines that can handle ONNX, MNN, and TensorRT models.
    This class provides methods for loading models, preprocessing inputs, and postprocessing outputs.
    Args:
        model_path (str): Path to the model file.
        width (int, optional): Width of the input images. If None, will be set from model input shape.
        height (int, optional): Height of the input images. If None, will be set from model input shape.
    """

    def __init__(
        self,
        model_path,
        width=None,
        height=None,
        *args,
        **kwargs,
    ):
        """
        model_path (string): path of the model file
        """
        logging.info(f"Instanciate InferenceEngine for model file: {model_path}")
        self.model_path = model_path
        self.input_shape = None
        self.batch_size = None
        self.channels = None
        self.height = height
        self.width = width
        self.model_type = self.__detect_model_type()
        self.__load_engine()
        logging.info(
            f"Model shape: Batch = {self.batch_size}, Channels = {self.channels}, Height = {self.height}, Width = {self.width}"
        )

    def __del__(self):
        logging.debug(
            f"Remove InferenceEngine instance for model file: {self.model_path}"
        )
        del self.engine

    def __detect_model_type(self):
        try:
            return ModelType(os.path.splitext(self.model_path)[1])
        except ValueError as e:
            raise NotImplementedError(e.args[0])

    def __load_engine(self):
        logging.info(f"Loading model {self.model_path}")
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path {self.model_path} does not exist")

        if self.model_type is ModelType.ONNX:
            if not onnxruntime_loaded:
                log.exception(f"Cannot import onnxruntime")
                raise ImportError("ONNXRuntime is not installed. Please install it.")
            logging.info("Using ONNXRuntime for inference")
            self._load_onnx_engine()

        elif self.model_type is ModelType.MNN:
            if not mnn_loaded:
                log.exception(f"Cannot import MNN")
                raise ImportError("MNN is not installed. Please install it.")
            logging.info("Using MNN for inference")
            self._load_mnn_engine()

        elif self.model_type is ModelType.TRT:
            if not trt_loaded:
                log.exception(f"Cannot import TensorRT")
                raise ImportError("TensorRT is not installed. Please install it.")
            if not pycuda_loaded:
                log.exception(f"Cannot import PyCUDA")
                raise ImportError("PyCUDA is not installed. Please install it.")
            logging.info("Using TensorRT for inference")
            self._load_trt_engine()
        else:
            raise NotImplementedError(
                f"Unsupported inference engine type '{self.model_type}' for now."
            )

    def _load_onnx_engine(self):
        providers = ["CUDAExecutionProvider"]
        logging.info(f"Use provider '{providers}'")
        self.engine = onnxruntime.InferenceSession(self.model_path, providers=providers)
        self.input_shape = self.engine.get_inputs()[0].shape
        self.batch_size = self.input_shape[0]
        self.channels = self.input_shape[1]
        if self.width is None:
            self.width = self.input_shape[3]
        if self.height is None:
            self.height = self.input_shape[2]

    def _load_mnn_engine(self):
        self.engine = MNN.Interpreter(self.model_path)
        self.session = self.engine.createSession()
        input_tensor = self.engine.getSessionInput(self.session)
        self.input_shape = input_tensor.getShape()
        self.batch_size = self.input_shape[0]
        self.channels = self.input_shape[1]
        if self.width is None:
            self.width = self.input_shape[3]
        if self.height is None:
            self.height = self.input_shape[2]

    def _load_trt_engine(self):
        import pycuda.autoinit

        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        f = open(self.model_path, "rb")
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)
        self.batch_size = self.input_shape[0]
        self.channels = self.input_shape[1]
        if self.width is None:
            self.width = self.input_shape[3]
        if self.height is None:
            self.height = self.input_shape[2]

    @abstractmethod
    def preprocess(self, img):
        pass

    @abstractmethod
    def postprocess(self, outputs):
        pass

    def predict(self, img):
        add_padding = 0
        if self.batch_size is not None and not isinstance(self.batch_size, str):
            if isinstance(img, list) and len(img) < self.batch_size:
                add_padding = self.batch_size - len(img)
                logging.debug(
                    f"Padding input with {add_padding} empty images to reach {self.batch_size} images"
                )
                img.extend([np.zeros_like(img[0])] * add_padding)
        x = self.preprocess(img)
        outputs = self.infer(x)
        predictions = self.postprocess(outputs)
        if add_padding > 0:
            predictions = predictions[:-add_padding]
        return predictions

    def __call__(self, img):
        return self.predict(img)

    def infer(self, image_tensor):
        """
        image_tensor (np array): NHWC tensor
        """
        if self.model_type is ModelType.ONNX:
            return self.__infer_onnx(image_tensor)
        elif self.model_type is ModelType.MNN:
            return self.__infer_mnn(image_tensor)
        elif self.model_type is ModelType.TRT:
            return self.__infer_trt(image_tensor)

    def __infer_onnx(self, image_tensor):
        image_tensor = np.transpose(image_tensor, (0, 3, 1, 2))
        input_name = self.engine.get_inputs()[0].name
        output = self.engine.run(None, {input_name: image_tensor})
        return output

    def __infer_mnn(self, image_tensor):
        input_tensor = self.engine.getSessionInput(self.session)
        image_tensor = np.asarray(image_tensor, dtype=np.float32)
        image_tensor = np.transpose(image_tensor, (0, 3, 1, 2))
        tmp_input = MNN.Tensor(
            image_tensor.shape,
            MNN.Halide_Type_Float,
            image_tensor,
            input_tensor.getDimensionType(),
        )

        input_tensor.copyFromHostTensor(tmp_input)

        self.engine.runSession(self.session)

        output_tensors = self.engine.getSessionOutputAll(self.session)
        tensor_list = []
        for key, tmp_tensor in output_tensors.items():
            if tmp_tensor.getDataType() == MNN.Halide_Type_Int:  # is
                np_type = np.int32
            if tmp_tensor.getDataType() == MNN.Halide_Type_Float:  # is
                np_type = np.float32
            tensor = MNN.Tensor(
                tmp_tensor.getShape(),
                tmp_tensor.getDataType(),
                np.zeros(tmp_tensor.getShape(), np_type),
                MNN.Tensor_DimensionType_Caffe,
            )

            tmp_tensor.copyToHostTensor(tensor)
            shape = tensor.getShape()
            data = tensor.getData()
            numpy_tensor = np.asarray(data)
            numpy_tensor = np.reshape(numpy_tensor, shape)
            tensor_list.append(numpy_tensor)

        return tensor_list

    def __infer_trt(self, image_tensor):
        image_tensor = np.asarray(image_tensor, dtype=np.float32)
        image_tensor = np.transpose(image_tensor, (0, 3, 1, 2))
        inputs = []
        outputs = []
        bindings = []
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        for i, binding in enumerate(self.engine):
            binding_shape = self.engine.get_binding_shape(binding)
            binding_shape[0] = image_tensor.shape[0]  # for dynamic batch
            size = trt.volume(binding_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(i, binding_shape)  # for dynamic batch
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))
        np.copyto(inputs[0][0], image_tensor.ravel())
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        pred = []
        for i, out in enumerate(outputs):
            out_shape = self.engine.get_binding_shape(self.engine[i + 1])
            out_shape[0] = image_tensor.shape[0]
            out = np.reshape(out[0], out_shape)
            pred.append(out)
        return pred
