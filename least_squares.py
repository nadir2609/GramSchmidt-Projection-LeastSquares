import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

# --- Helper Functions for Pretty Printing ---
def print_matrix(name, m):
    """Prints a matrix or vector with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)

# ====================================================================
# PART 3: LEAST SQUARES, MODEL FITTING, AND VISUALIZATION
# ====================================================================

print("="*60)
print("PART 5.3: LEAST SQUARES, MODEL FITTING, AND VISUALIZATION")
print("="*60)

# --- Problem Setup: Multi-feature House Price Data ---
np.random.seed(0)
num_houses = 20
house_sizes_sq_m = np.linspace(80, 300, num_houses)
house_ages_years = np.linspace(1, 25, num_houses)

true_prices = 80 + 1.8 * house_sizes_sq_m - 2.5 * house_ages_years + 0.005 * house_sizes_sq_m**2
noise = np.random.normal(0, 30, house_sizes_sq_m.shape)
observed_prices = true_prices + noise

x1_feature_size = house_sizes_sq_m
x2_feature_age = house_ages_years
b_prices = observed_prices

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def solve_system(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    Returns:
        - list: solution vector x if unique solution exists
        - str: 'No solution' if inconsistent
        - str: 'Infinite solutions' if system has free variables
    """
    n = len(A)
    augmented = copy.deepcopy(A)
    for i in range(n):
        augmented[i].append(b[i])

    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        max_value = abs(augmented[i][i])
        for r in range(i + 1, n):
            if abs(augmented[r][i]) > max_value:
                max_value = abs(augmented[r][i])
                max_row = r

        # Swap rows if needed
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

        # Skip near-zero pivots
        if abs(augmented[i][i]) < 1e-12:
            continue

        # Eliminate below
        for j in range(i + 1, n):
            if augmented[j][i] != 0:
                factor = augmented[j][i] / augmented[i][i]
                for k in range(i, n + 1):
                    augmented[j][k] -= factor * augmented[i][k]

    # Check for inconsistency
    for row in augmented:
        if all(abs(val) < 1e-12 for val in row[:-1]) and abs(row[-1]) > 1e-12:
            return "No solution"

    # Count rank
    rank = sum(not all(abs(val) < 1e-12 for val in row[:-1]) for row in augmented)
    if rank < n:
        return "Infinite solutions"

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        if abs(augmented[i][i]) < 1e-12:
            return "Infinite solutions"
        total = sum(augmented[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (augmented[i][-1] - total) / augmented[i][i]

    return x


def transpose_matrix(M):
    """Computes the transpose of a matrix M."""
    rows = len(M)
    cols = len(M[0])
    M_T = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            M_T[j][i] = M[i][j]
    return M_T


def multiply_matrices(M1, M2):
    """Computes the product of two matrices M1 and M2."""
    A = np.array(M1, dtype=float)
    B = np.array(M2, dtype=float)
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        print("Wrong matrix dimensions for multiplication.")

    result = np.zeros((rows_A, cols_B), dtype=float)
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]

    return result

# ====================================================================
# Part 5.3.1: Least Squares Solver from Scratch
# ====================================================================

def least_squares(A, b):
    """
    Solve the least squares problem min_x ||Ax - b||^2 using ormal equations:
        (A^T A) x = A^T b
    Returns x as a column numpy array.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    A_T = transpose_matrix(A.tolist())
    ATA = multiply_matrices(A_T, A)
    ATb = multiply_matrices(A_T, b)

    ATA_list = ATA.tolist()
    ATb_list = [float(val) for val in ATb.flatten()]

    x_list = solve_system(ATA_list, ATb_list)

    if isinstance(x_list, str):
        x_np = np.linalg.lstsq(A, b, rcond=None)[0]
        return x_np.reshape(-1, 1)

    x = np.array(x_list, dtype=float).reshape(-1, 1)
    return x

# ====================================================================
# Part 5.3.2: Application: Multiple Linear Housing Model
# ====================================================================
print("\n--- Part 5.3.2: Application: Multiple Linear Housing Model ---")

# 1. Construct the feature matrix A_linear for the model y = c0 + c1*size + c2*age
col_ones = np.ones_like(x1_feature_size)
A_linear = np.column_stack([col_ones, x1_feature_size, x2_feature_age])

print_matrix("Linear Feature Matrix A_linear (first 5 rows)", A_linear[:5])

# 2. Use your least_squares function to find the optimal weights
x_hat_linear = least_squares(A_linear, b_prices)
print_matrix("Optimal Weights x_hat_linear [c0, c1, c2] (from scratch)", x_hat_linear)

# 3. Verify your result with numpy.linalg.lstsq
print("--- Verifying with NumPy ---")
x_hat_linear_np = np.linalg.lstsq(A_linear, b_prices, rcond=None)[0]
print_matrix("Optimal Weights x_hat_linear [c0, c1, c2] (from NumPy)", x_hat_linear_np)

# 4. Interpretation
if x_hat_linear is not None:
    c0, c1, c2 = x_hat_linear.flatten()
    size_to_predict = 200
    age_to_predict = 5
    predicted_price = c0 + c1 * size_to_predict + c2 * age_to_predict
    print(f"\n--- Interpretation ---")
    print(f"The linear model is: price = {c0:.2f} + {c1:.2f}*size + {c2:.2f}*age")
    print(f"Predicted price for a {size_to_predict} sq m, {age_to_predict}-year-old house: ${predicted_price:.2f}k")
else:
    print("\nCannot make prediction, weights not calculated.")

# ====================================================================
# Part 5.3.3: Application: Polynomial Housing Model
# ====================================================================
print("\n--- Part 5.3.3: Application: Polynomial Housing Model ---")

# 1. Construct the feature matrix A_poly for y = c0 + c1*size + c2*age + c3*size^2
size_squared = x1_feature_size**2
A_poly = np.column_stack([col_ones, x1_feature_size, x2_feature_age, size_squared])
print_matrix("Polynomial Feature Matrix A_poly (first 5 rows)", A_poly[:5])

# 2. Use your *same* least_squares function to find the new optimal weights
x_hat_poly = least_squares(A_poly, b_prices)
print_matrix("Optimal Weights x_hat_poly [c0, c1, c2, c3] (from scratch)", x_hat_poly)

# ====================================================================
# Part 5.3.4: Visualization and Comparison
# ====================================================================
print("\n--- Part 5.3.4: Visualization and Comparison (3D Plot) ---")

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the raw data points
ax.scatter(x1_feature_size, x2_feature_age, b_prices, c='r', marker='o', label='Actual Data Points')

# Create a mesh grid to plot the model surfaces
size_surf, age_surf = np.meshgrid(
    np.linspace(x1_feature_size.min(), x1_feature_size.max(), 20),
    np.linspace(x2_feature_age.min(), x2_feature_age.max(), 20)
)

# Plot the linear model fit (a plane)
if x_hat_linear is not None:
    c0_lin, c1_lin, c2_lin = x_hat_linear.flatten()
    price_surf_linear = c0_lin + c1_lin * size_surf + c2_lin * age_surf
    ax.plot_surface(size_surf, age_surf, price_surf_linear, color='cyan', alpha=0.5)

# Plot the polynomial model fit (a curved surface)
if x_hat_poly is not None:
    c0_poly, c1_poly, c2_poly, c3_poly = x_hat_poly.flatten()
    price_surf_poly = c0_poly + c1_poly * size_surf + c2_poly * age_surf + c3_poly * size_surf**2
    ax.plot_surface(size_surf, age_surf, price_surf_poly, color='magenta', alpha=0.5)

ax.set_xlabel('House Size (sq m)')
ax.set_ylabel('House Age (years)')
ax.set_zlabel('Price (in thousands of $)')
ax.set_title('Housing Price vs. Size and Age: Model Comparison')

print("Visualizing models: Cyan surface is Linear, Magenta surface is Polynomial.")
plt.show()

# Answer to analysis question:

# Adding high degree polynomial features can lead to overfitting. 
# It is when model fits training data well
# but perform bad on test data
# Overfitting models have very high variance:
# small changes in the data leads to big changes in predictions.