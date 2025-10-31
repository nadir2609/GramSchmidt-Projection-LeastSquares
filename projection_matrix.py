import numpy as np

# --- Helper Functions for Pretty Printing ---
def print_vectors(name, vecs):
    """Prints a list of vectors, handling both lists of arrays and 2D arrays."""
    print(f"{name}:")
    if vecs is None or len(vecs) == 0:
        print("[] (or not implemented)")
    # Check if it's a list of 1D arrays
    elif isinstance(vecs, list) and all(isinstance(v, np.ndarray) for v in vecs):
        for i, v in enumerate(vecs):
            np.set_printoptions(precision=4, suppress=True)
            print(f"  Vector {i+1}:\n{v.reshape(-1, 1)}")
    else:
        print("Unsupported format for printing vectors.")
    print("-" * 40)

def print_matrix(name, m):
    """Prints a matrix with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)
# --- Setup ---
A2 = np.array([
    [1., 0.],
    [1., 1.],
    [1., 2.]
])
b2 = np.array([6., 0., 0.])
print_matrix("Matrix A for Part 5.2", A2)
print_matrix("Vector b for Part 5.2", b2.reshape(-1, 1))

# --- Reusable Helper Functions (Students must implement or reuse) ---

def transpose_matrix(M):
    """Computes the transpose of a matrix M."""
    rows = len(M)
    cols = len(M[0])
    M_T = []  

    for j in range(cols):       
        row = []                
        for i in range(rows):   
            row.append(0)
        M_T.append(row)        
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

    # Check that inner dimensions match 
    if cols_A != rows_B:
        print("Wrong matrix dimensions for multiplication.")

    # Prepare result matrix of zeros with correct shape
    result = np.zeros((rows_A, cols_B), dtype=float)

    # Compute product with explicit nested loops 
    for i in range(rows_A):
        for j in range(cols_B):
            sum_val = 0
            for k in range(cols_A):
                sum_val += A[i, k] * B[k, j]
            result[i, j] = sum_val

    return result

# --- 5.2.1: The Projection Matrix ---

def create_projection_matrix(A):
    """
    Creates a projection matrix P that projects onto the column space of A.
    Formula: P = A(A^T A)^-1 A^T
    """
    A_T = transpose_matrix(A)
    A_T_A = multiply_matrices(A_T, A)
    A_T_A_inv = np.linalg.inv(A_T_A)
    P = multiply_matrices(multiply_matrices(A, A_T_A_inv), A_T)
    
    return P

print("\n--- 5.2.1: Creating the Projection Matrix ---")
P = create_projection_matrix(A2)
print_matrix("Projection Matrix P (from scratch)", P)


# --- 5.2.2: Projecting the Data Vector ---
print("\n--- 5.2.2: Projecting the Vector ---")
if P is not None:
    # p = P @ b2
    p = multiply_matrices(P, b2.reshape(-1, 1)).flatten() if P is not None else None
    print_matrix("Projected vector p = Pb", p.reshape(-1, 1) if p is not None else None)
    if p is not None:
        print("These are the y-values of the best-fit line at t=0, 1, 2.")
else:
    print("Projection matrix not implemented.")


# --- 5.2.3: Decomposing the Vector and Verifying Orthogonality ---
print("\n--- 5.2.3: Verifying Orthogonality ---")
if p is not None:
    # 1. Calculate the error vector e = b - p
    e = b2 - p
    print_matrix("Error vector e = b - p", e.reshape(-1, 1))

    # 2. Verification 1: p and e must be orthogonal
    p_dot_e = np.dot(p, e)
    print(f"Verification 1: Dot product of p and e = {p_dot_e:.4f} (should be 0)")
    print(f"Are they orthogonal? {np.isclose(p_dot_e, 0)}\n")

    # 3. Verification 2: e must be in the left nullspace of A
    #    This means A^T @ e should be the zero vector.
    AT2 = A2.T # Using numpy's transpose for verification step
    AT_e = AT2 @ e
    print("Verification 2: e must be in the Left Nullspace (A^T @ e = 0)")
    print_matrix("A^T @ e", AT_e.reshape(-1, 1))
    print(f"Is A^T @ e close to zero? {np.allclose(AT_e, 0)}")
else:
    print("Projected vector p not available, cannot verify.")
