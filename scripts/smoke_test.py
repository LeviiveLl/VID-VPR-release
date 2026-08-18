import argparse

import torch

from vid_vpr import load_vid_vpr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/student.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    model = load_vid_vpr(args.checkpoint).to(args.device).eval()
    images = torch.randn(1, 3, args.size, args.size, device=args.device)
    with torch.inference_mode():
        descriptors = model(images)
    print("shape:", tuple(descriptors.shape))
    print("norm:", descriptors.norm(dim=-1).tolist())


if __name__ == "__main__":
    main()
