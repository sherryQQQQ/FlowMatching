#!/usr/bin/env python3
"""
Example script to run training with loss tracking
"""

import subprocess
import sys
import os

def main():
    """Run training with loss tracking enabled"""
    
    # Create workdir if it doesn't exist
    workdir = './workdir'
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(workdir, 'checkpoints'), exist_ok=True)
    
    # Training command
    cmd = [
        sys.executable, 'main.py',
        '--mode', 'train',
        '--workdir', workdir,
        '--n_epochs', '50'  # Adjust as needed
    ]
    
    print("🚀 Starting training with loss tracking...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Loss curves will be saved to: {workdir}/loss_curves.png")
    print("=" * 60)
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print(f"📊 Check loss curves at: {workdir}/loss_curves.png")
        print(f"📁 All outputs saved in: {workdir}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()
