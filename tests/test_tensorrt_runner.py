import tempfile
import unittest
from pathlib import Path

from g1_playground.policy.tensorrt_runner import TensorRTRunner, _model_digest


class TestTensorRTModelIdentity(unittest.TestCase):
    def test_external_weights_are_part_of_the_engine_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "policy.onnx"
            weights = Path(directory) / "policy.onnx.data"
            model.write_bytes(b"graph")
            weights.write_bytes(b"weights-a")
            first = _model_digest(model)

            weights.write_bytes(b"weights-b")

            self.assertNotEqual(first, _model_digest(model))

    def test_graph_changes_are_part_of_the_engine_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "policy.onnx"
            model.write_bytes(b"graph-a")
            first = _model_digest(model)

            model.write_bytes(b"graph-b")

            self.assertNotEqual(first, _model_digest(model))

    def test_only_existing_onnx_files_reach_tensorrt(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "policy.pt"
            model.write_bytes(b"not an ONNX graph")

            with self.assertRaisesRegex(ValueError, "requires an ONNX model"):
                TensorRTRunner(model)


if __name__ == "__main__":
    unittest.main()
