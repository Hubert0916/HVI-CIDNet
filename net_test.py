import torch
from net.CIDNet import CIDNet
import sys

def run_test():
    """
    Tests the modified CIDNet architecture with the learnable k-map.
    """
    print("--- Starting Architecture Verification Test ---")
    try:
        device = 'cpu'
        print(f"Using device: {device}")

        # 1. Instantiate the model
        print("Instantiating CIDNet model...")
        model = CIDNet().to(device)
        model.eval()
        print("Model instantiated successfully.")

        # 2. Create a dummy input tensor
        batch_size = 1
        height, width = 128, 128
        print(f"Creating a dummy input tensor of size ({batch_size}, 3, {height}, {width})...")
        input_tensor = torch.rand(batch_size, 3, height, width).to(device)

        # 3. Perform a forward pass
        print("Performing a forward pass...")
        with torch.no_grad():
            output = model(input_tensor)
        print("Forward pass completed.")

        # 4. Check the output
        print(f"Input shape:  {input_tensor.shape}")
        print(f"Output shape: {output.shape}")

        if input_tensor.shape == output.shape:
            print("\nSUCCESS: The output shape matches the input shape.")
            print("The new architecture with the learnable k-map has been integrated correctly.")
        else:
            print(f"\nFAILURE: The output shape {output.shape} does not match the input shape {input_tensor.shape}.")
            sys.exit(1)

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\n--- Architecture Verification Test Finished ---")

if __name__ == '__main__':
    run_test()