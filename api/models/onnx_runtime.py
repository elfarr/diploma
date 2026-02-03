from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import onnxruntime as ort


@dataclass
class OnnxModel:
    session: ort.InferenceSession
    input_name: str
    output_name: str

    @staticmethod
    def load(model_path: str) -> "OnnxModel":
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        outputs = sess.get_outputs()
        prob_out = None
        for o in outputs:
            if "prob" in o.name or (len(o.shape) == 2):
                prob_out = o.name
                break
        out = prob_out or outputs[0].name
        return OnnxModel(session=sess, input_name=inp, output_name=out)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: x})
        return outputs[0]
