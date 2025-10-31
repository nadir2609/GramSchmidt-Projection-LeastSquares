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


# --- Problem Setup ---
# A set of linearly independent vectors to be orthonormalized
v1 = np.array([1., -1., 1.])
v2 = np.array([1., 0., 1.])
v3 = np.array([1., 1., 2.])

input_vectors = [v1, v2, v3]

print_vectors("Input Vectors (Linearly Independent)", input_vectors)

def projection (u, v):
    return (u.dot(v) / v.dot(v)) * v

# ====================================================================
# Part 5.1.1: Gram-Schmidt Process from Scratch
# ====================================================================

def gram_schmidt(vectors):
    n = len(vectors)
    orthonormal_basis = []
    for i in range(n):
        u = vectors[i]
        for j in range(len(orthonormal_basis)):
            q = orthonormal_basis[j]
            u = u - projection(u, q)
        u = u / np.linalg.norm(u)
        orthonormal_basis.append(u)
    return orthonormal_basis


# --- Calling the function for Part 2.1 ---
print("--- Part 2.1: Applying the Gram-Schmidt Process ---")
orthonormal_vectors = gram_schmidt(input_vectors)
print_vectors("Orthonormal Basis (from scratch)", orthonormal_vectors)


# ====================================================================
# Part 5.1.2: Verification
# ====================================================================
print("--- Part 2.2: Verification ---")

if not orthonormal_vectors:
    print("Orthonormal basis not implemented, cannot perform verification.")
else:
    # 1. Verification: Orthogonality
    print("--- Verifying Orthogonality ---")
    is_orthogonal = True
    num_vectors = len(orthonormal_vectors)
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            v1 = orthonormal_vectors[i]
            v2 = orthonormal_vectors[j]
            dot_product = np.dot(v1, v2)
            print(f"Dot product of Vector {i+1} and Vector {j+1}: {dot_product:.6f}")
            if not np.isclose(dot_product, 0):
                is_orthogonal = False
    print(f"Are all distinct pairs orthogonal? {is_orthogonal}\n")

    # 2. Verification: Normalization
    print("--- Verifying Normalization ---")
    is_normalized = True
    for i, v in enumerate(orthonormal_vectors):
        norm = np.linalg.norm(v)
        print(f"Norm of Vector {i+1}: {norm:.6f}")
        if not np.isclose(norm, 1):
            is_normalized = False
    print(f"Are all vectors normalized (unit vectors)? {is_normalized}\n")

    # 3. Verification: NumPy Comparison using QR Decomposition
    print("--- Verifying with NumPy's QR Decomposition ---")
    # Stack the original vectors as columns of a matrix A
    A = np.column_stack(input_vectors)
    print_matrix("Matrix A (from input vectors)", A)
    
    # Perform QR decomposition
    Q, R = np.linalg.qr(A)
    
    print_matrix("Q Matrix from NumPy's np.linalg.qr(A)", Q)
    print("The columns of this Q matrix form NumPy's orthonormal basis.")
    print("Your basis should be equivalent (individual vectors may have opposite signs).")
