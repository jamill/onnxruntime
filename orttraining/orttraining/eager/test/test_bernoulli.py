import onnxruntime_pybind11_state as torch_ort
import torch

torch_ort.set_default_logger_severity(0)
torch_ort.set_default_logger_verbosity(4)


device = torch_ort.device()
cpu_tensor = torch.rand(5)
ort_tensor = cpu_tensor.to(device)

cpu_result = torch.bernoulli(cpu_tensor, 0.5)
print(cpu_result)

ort_result = torch.bernoulli(ort_tensor, 0.5)
print(ort_result)
