import torch

def check_pytorch_gpu():
    print("PyTorch Version:", torch.__version__)
    gpu_available = torch.cuda.is_available()
    print("GPU Available:", gpu_available)
    if gpu_available:
        device = torch.device("cuda")
        print("GPU Device Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)

        # Create a random tensor and move it to GPU
        tensor = torch.randn(3, 3).to(device)
        print("Tensor on GPU:\n", tensor)

        # Perform a simple computation on GPU
        result = tensor @ tensor  # Matrix multiplication
        print("Computation on GPU (Tensor @ Tensor):\n", result)

        # Check if the tensor is actually on the GPU
        if result.device.type == "cuda":
            print("GPU test successful! Computation performed on:", result.device)
        else:
            print("GPU test failed. Tensor is not on GPU.")

if __name__ == "__main__":
    check_pytorch_gpu()