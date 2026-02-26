from os import path as osp
from typing import List, Optional, Union

import numpy as np
import torch

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoprimaryctx

from utils.img_util import img2tensor, tensor2img

from config.derain import DERAIN_ENGINE

Frame = np.ndarray
Frames = Union[List[Frame], np.ndarray]
_DERAIN_SINGLETON: Optional["DerainTRT"] = None

def deraining(frames: Frames) -> Frames:
    return _get_singleton().apply_batch(frames)


def _get_singleton() -> "DerainTRT":
    global _DERAIN_SINGLETON
    if _DERAIN_SINGLETON is None:
        _DERAIN_SINGLETON = DerainTRT()
    return _DERAIN_SINGLETON


class DerainTRT:
    def __init__(self) -> None:
        # Engine path comes from config.
        self.engine_path = DERAIN_ENGINE
        if not osp.isfile(self.engine_path):
            raise FileNotFoundError(
                f"TensorRT engine not found: {self.engine_path}. "
                "Check config/derain.py (DERAIN_ENGINE)."
            )
        self._load_engine(self.engine_path)
        self._h_input = None
        self._h_output = None
        self._d_input = None
        self._d_output = None
        self._last_input_shape = None
        self._last_output_shape = None
        self._ctx = None

    def _load_engine(self, engine_path: str) -> None:
        # Load TensorRT engine once.
        cuda.init()
        dev = cuda.Device(0)
        self._ctx = dev.retain_primary_context()
        self._ctx.push()
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def apply_batch(self, frames: Frames) -> Frames:
        if isinstance(frames, list):
            return [self.apply(f) for f in frames]

        if isinstance(frames, np.ndarray):
            if frames.ndim == 3:
                print("im in 3dim!")
                return self.apply(frames)
            if frames.ndim == 4:
                print("im in 4-dim! i can handle batch!")
                print("frame shape: ", frames.shape)
                b, h, w, c = frames.shape
                
                if c != 3:
                    raise ValueError(f"Expected 3 channels, got {c}")
                inp = frames.astype(np.float32) / 255.0
                print("type 변환된 frame: ", inp.shape)
                # BGR -> RGB
                inp = inp[..., ::-1]
                # NHWC -> NCHW
                inp = np.transpose(inp, (0, 3, 1, 2))
                print("transpose 된 frame: ", inp.shape)
                output = self._infer(inp)
                if output.ndim == 1:
                    print("why am i here?")
                    output = output.reshape((b, 3, h, w))
                # NCHW -> NHWC, RGB -> BGR, float -> uint8
                out = np.transpose(output, (0, 2, 3, 1))
                out = np.clip(out, 0.0, 1.0)
                out = out[..., ::-1]
                out = (out * 255.0).round().astype(np.uint8)
                return out

        raise TypeError(
            f"Unsupported frames type/shape: {type(frames)} {getattr(frames, 'shape', None)}"
        )

    def apply(self, frame: Frame) -> Frame:
        # (h,w,c) = (1024, 1024, 3) -> (c, h, w) = (3, 1024, 1024)
        inp = img2tensor(frame.astype(np.float32) / 255.0, bgr2rgb=True, float32=True)
        # (3, 1024, 1024) -> (1, 3, 1024, 1024)
        inp_np = inp.unsqueeze(dim=0).numpy()
        # Run TRT inference
        output_flat = self._infer(inp_np)

        h, w = frame.shape[:2]
        if output_flat.ndim == 1:
            out = output_flat.reshape((1, 3, h, w))
        else:
            out = output_flat
        # CHW -> HWC uint8.
        denoised_roi = tensor2img([torch.from_numpy(out)])
        return denoised_roi

    def _infer(self, img: np.ndarray) -> np.ndarray:
        input_name = self.engine.get_tensor_name(0)
        output_name = self.engine.get_tensor_name(1)
        input_dtype = trt.nptype(self.engine.get_tensor_dtype(input_name))
        output_dtype = trt.nptype(self.engine.get_tensor_dtype(output_name))

        self.context.set_input_shape(input_name, img.shape)
        output_shape = tuple(self.context.get_tensor_shape(output_name))

        # Allocate host/device buffers once and reuse (realloc if shape changes).
        if (self._h_input is None or
                self._last_input_shape != img.shape or
                self._last_output_shape != output_shape):
            self._h_input = cuda.pagelocked_empty(
                trt.volume(img.shape), dtype=input_dtype
            )
            self._h_output = cuda.pagelocked_empty(
                trt.volume(output_shape), dtype=output_dtype
            )
            self._d_input = cuda.mem_alloc(self._h_input.nbytes)
            self._d_output = cuda.mem_alloc(self._h_output.nbytes)
            self._last_input_shape = img.shape
            self._last_output_shape = output_shape

        if img.dtype != input_dtype:
            img = img.astype(input_dtype, copy=False)
        np.copyto(self._h_input, img.ravel())

        self.context.set_tensor_address(input_name, self._d_input)
        self.context.set_tensor_address(output_name, self._d_output)

        cuda.memcpy_htod_async(self._d_input, self._h_input, self.stream)
        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if ok is False:
            raise RuntimeError(
                "TensorRT execution failed (execute_async_v3 returned False)."
            )
        cuda.memcpy_dtoh_async(self._h_output, self._d_output, self.stream)
        self.stream.synchronize()
        return self._h_output.reshape(output_shape)
