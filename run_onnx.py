import onnx
import onnxruntime
import torch

onnx_path = "./young-car.onnx"
if __name__ == "__main__":
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    dummy_input = torch.FloatTensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    print("============================================")
    print("states")
    print(dummy_input)
    print("action")
    print(ort_outs)
    print("============================================")

    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")