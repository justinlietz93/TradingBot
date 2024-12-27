import subprocess

def install_dependencies():
    # Activate the trading_env virtual environment
    subprocess.call(["trading_env\\Scripts\\activate.bat"])

    # Install dependencies from requirements.txt
    subprocess.call(["pip", "install", "--exists-action", "i", "-r", "requirements.txt"])

    # Deactivate the virtual environment
    subprocess.call(["deactivate"])

if __name__ == "__main__":
    install_dependencies()
