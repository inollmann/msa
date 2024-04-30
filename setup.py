from distutils.core import setup
import py2exe

setup(
    name="YourPackageName",
    version="1.0",
    packages=['model', 'utils'],  # Explicitly include 'model' and 'utils' packages
    # Alternatively, you can use the 'packages' option to automatically include packages
    # options={"py2exe": {"packages": ["model", "utils"]}},
    # Add other options as needed
    console=["your_script.py"],  # Entry point of your application
)

