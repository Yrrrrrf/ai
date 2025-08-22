# scripts/some.py
# This script is a placeholder to demonstrate how to run a simple script.


import torch

def main():
    print("\033[H\033[J", end="")
    print("\033[1;32mSome script running...\033[0m")

    t = torch.cuda.is_available()
    print(t)

        
if __name__ == "__main__":
    main()
