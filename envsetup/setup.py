import subprocess, sys
import os
import sys
from typing import List

def setup_env():
    # Check if the virtual environment is already activated
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # make sure requirements.txt is installed in the current environment
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '-r', 'requirements.txt'])
        return

    # Create a virtual environment
    os.system('python -m venv venv -r requirements.txt')

    # Activate the virtual environment
    if os.name == 'nt':  # For Windows
        activate_script = os.path.join('venv', 'Scripts', 'activate')
    else:  # For Unix/Linux/Mac
        activate_script = os.path.join('venv', 'bin', 'activate')

    print(f"To activate the virtual environment, run: source {activate_script}")
    


    print('All dependencies ready.')
def set_up_google_drive():
    from google.colab import drive
    drive.mount('/content/drive')