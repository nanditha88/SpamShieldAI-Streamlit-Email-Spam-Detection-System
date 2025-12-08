# setup.py - Setup script for email spam detection
import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = [
        'streamlit',
        'pandas',
        'numpy', 
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'wordcloud'
    ]
    
    print("Installing required packages...")
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("\nAll packages installed successfully!")
    print("\nTo run the application, execute:")
    print("streamlit run app.py")

if __name__ == "__main__":
    install_packages()