import launch
import importlib
from packaging.version import Version
from packaging.requirements import Requirement

def is_installed(pip_package):
    """
    Check if a package is installed and meets version requirements specified in pip-style format.

    Args:
        pip_package (str): Package name in pip-style format (e.g., "numpy>=1.22.0").

    Returns:
        bool: True if the package is installed and meets the version requirement, False otherwise.
    """
    try:
        # Parse the pip-style package name and version constraints
        requirement = Requirement(pip_package)
        package_name = requirement.name
        specifier = requirement.specifier  # e.g., >=1.22.0

        # Check if the package is installed
        dist = importlib.metadata.distribution(package_name)
        installed_version = Version(dist.version)

        # Check version constraints
        if specifier.contains(installed_version):
            return True
        else:
            print(f"Installed version of {package_name} ({installed_version}) does not satisfy the requirement ({specifier}).")
            return False
    except importlib.metadata.PackageNotFoundError:
        print(f"Package {pip_package} is not installed.")
        return False


# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith(";")]

# Add webui packages
webui_packages = [
    "gradio",
    "fastapi",
    "uvicorn",
    "python-multipart"
]
requirements.extend(webui_packages)

for req in requirements:
    if not is_installed(req):
        launch.run_pip(f"install {req}", f"sd-webui-lsnet requirement: {req}")