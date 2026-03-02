# Machine Learning of Differential Equations using Gaussian Processes

This project explores how Gaussian Processes (GPs) can be used to solve and analyze differential equations, including regression, kernel design, and physics-informed learning.

## Project Structure

- **Chapter 2 Gaussian Processes/**: Introduction to GPs, sampling, and practical recommendations.
- **Chapter 3 Regression/**: GP regression experiments, error analysis, and computation time studies.
- **Chapter 4 Physics-Informed GP/**: Applying GPs to solve partial differential equations (PDEs) with physics-informed kernels.

## Key Files

- `2.1.3_Sample_Paths.py`: Draws sample paths from different GP kernels.
- `2.2.2_Practical_Recommendations.py`: Practical tips for GP computation.
- `3.5.1_GP_using_different_kernels.py`: GP regression with various kernels.
- `3.5.2_Error_vs_dimension.py`: Analyzes GP regression error as input dimension increases.
- `3.5.2_computation_time_vs_dimension.py`: Measures computation time for GP regression in higher dimensions.
- `4.3.2_Two_dim_linear_PDE.py`: Physics-informed GP for a 2D linear PDE.
- `heat_equation.py`: Contains exact solutions and data generation for the heat equation.

## How to Run

1. **Set up the environment:**
   - Use the provided Python virtual environment (`venv`).
   - Install dependencies: `pip install numpy matplotlib scipy sympy`

2. **Run scripts:**
   - Example: `python Chapter\ 2\ Gaussian\ Processes/2.1.3_Sample_Paths.py`
   - For other scripts, adjust the path as needed.

## Requirements
- Python 3.8+
- numpy
- matplotlib
- scipy
- sympy

## Notes
- Some scripts require a graphical interface to display plots.
- For physics-informed GP scripts, symbolic computation is used (via `sympy`).
- All code is organized by chapter for clarity.

## License
This project is for educational and research purposes.