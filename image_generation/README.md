# Diffusion Model - Installation Guide

This guide will help you set up the required environment and dependencies to run the diffusion model.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.8 or higher
- `conda` (Anaconda or Miniconda)
- `git` (optional, for cloning the repository)

## Installation Steps

1. **Clone the Repository** (if not already done):
    ```bash
    git clone https://github.com/your-username/ADLCV-project.git
    cd ADLCV-project/image_generation
    ```

2. **Create a Conda Environment**:
    Create and activate a new conda environment:
    ```bash
    conda create --name diffusers-env python=3.8 -y
    conda activate diffusers-env
    ```

3. **Install Required Dependencies**:
    Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4. **Verify Installation**:
    Test the setup to ensure all dependencies are installed correctly:
    ```bash
    python -c "import torch; print(torch.__version__)"
    ```

## Additional Notes

- If you encounter any issues during installation, ensure your conda and Python versions are up-to-date.
- For GPU acceleration, ensure you have CUDA installed and a compatible version of PyTorch.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Acknowledgments

Special thanks to the contributors and the open-source community for their support.
