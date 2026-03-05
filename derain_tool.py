import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoprimaryctx
from os import path as osp
from config.cfg import DERAIN_ENGINE, DERAIN_ROI_SIZE

_DERAIN_SINGLETON = None

def deraining(frames, roi_size=None):
    global _DERAIN_SINGLETON
    if _DERAIN_SINGLETON is None: _DERAIN_SINGLETON = DerainTRT()
    return _DERAIN_SINGLETON.apply(frames, roi_size or DERAIN_ROI_SIZE)

class DerainTRT:
    def __init__(self):
        cuda.init()
        self.ctx = cuda.Device(0).retain_primary_context()
        self.ctx.push()
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(DERAIN_ENGINE, "rb") as f, trt.Runtime(self.logger) as r:
            self.engine = r.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.buffs = [None] * 4 # h_in, h_out, d_in, d_out
        self.last_sh = (None, None)

    def apply(self, frames, roi_size=None):
        is_arr = isinstance(frames, np.ndarray)
        f_list = [frames] if (is_arr and frames.ndim == 3) else list(frames)
        
        if roi_size:
            out_fs = [f.copy() for f in f_list]
            crops, meta = [], []
            for i, f in enumerate(f_list):
                h, w = f.shape[:2]
                s = min(roi_size, h, w)
                x, y = (w-s)//2, (h-s)//2
                crops.append(f[y:y+s, x:x+s])
                meta.append((i, x, y, s))
            
            denoised = self._infer_batch(np.stack(crops))
            for d, (i, x, y, s) in zip(denoised, meta):
                out_fs[i][y:y+s, x:x+s] = d
            res = out_fs
        else:
            res = self._infer_batch(np.stack(f_list))

        return np.stack(res) if (is_arr and frames.ndim == 4) else (res[0] if is_arr else res)

    def _infer_batch(self, batch):
        # Pre-process: BGR->RGB, NHWC->NCHW, Normalize
        x = (batch[..., ::-1].transpose(0, 3, 1, 2) / 255.0).astype(np.float32)
        
        i_nm, o_nm = self.engine.get_tensor_name(0), self.engine.get_tensor_name(1)
        self.context.set_input_shape(i_nm, x.shape)
        o_sh = tuple(self.context.get_tensor_shape(o_nm))

        if self.last_sh != (x.shape, o_sh):
            self.buffs[0] = cuda.pagelocked_empty(x.size, np.float32)
            self.buffs[1] = cuda.pagelocked_empty(trt.volume(o_sh), np.float32)
            self.buffs[2] = cuda.mem_alloc(self.buffs[0].nbytes)
            self.buffs[3] = cuda.mem_alloc(self.buffs[1].nbytes)
            self.last_sh = (x.shape, o_sh)

        np.copyto(self.buffs[0], x.ravel())
        self.context.set_tensor_address(i_nm, self.buffs[2])
        self.context.set_tensor_address(o_nm, self.buffs[3])

        cuda.memcpy_htod_async(self.buffs[2], self.buffs[0], self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.buffs[1], self.buffs[3], self.stream)
        self.stream.synchronize()

        # Post-process: NCHW->NHWC, RGB->BGR, Denormalize
        out = self.buffs[1].reshape(o_sh).transpose(0, 2, 3, 1)
        return (np.clip(out, 0, 1)[..., ::-1] * 255).round().astype(np.uint8)
