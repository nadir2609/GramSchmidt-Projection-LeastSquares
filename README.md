---

```markdown
# Math4AI – Linear Algebra Assignment 5

This project implements three key linear algebra algorithms fundamental to AI and machine learning:

1. **Gram-Schmidt Process** – Generates an orthonormal basis from a set of linearly independent vectors.  
2. **Projection Matrix** – Constructs \( P = A(A^T A)^{-1}A^T \) to project vectors onto a subspace.  
3. **Least Squares Solver** – Solves \( \hat{x} = (A^T A)^{-1}A^T b \) to find the best-fit solution when no exact solution exists.

These implementations were coded **from scratch** using Python and NumPy to deepen understanding of how these mathematical tools connect to AI concepts like feature decorrelation, projection, and regression.

---

## 📁 Project Structure

```

env1/                     # Optional: your virtual environment folder
.gitignore
gram_schmidt.py           # Task 1: Gram-Schmidt implementation
projection_matrix.py      # Task 2: Projection Matrix computation
least_squares.py          # Task 3: Least Squares solver
README.md                 # Project description and setup guide
requirements.txt          # Required Python packages

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Math4AI-Linear-Algebra-Assignment5.git
cd Math4AI-Linear-Algebra-Assignment5
````

### 2. Create a Virtual Environment

```bash
python -m venv env1
```

### 3. Activate the Virtual Environment

* **Windows:**

  ```bash
  env1\Scripts\activate
  ```
* **macOS / Linux:**

  ```bash
  source env1/bin/activate
  ```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Each script can be run independently:

```bash
python gram_schmidt.py
python projection_matrix.py
python least_squares.py
```

These scripts demonstrate:

* **Orthonormalization (Gram-Schmidt)**
* **Vector projection onto subspaces**
* **Least squares regression for best-fit solutions**

---

## 🧠 Conceptual Links

* **Gram-Schmidt** → produces uncorrelated (orthogonal) feature sets
* **Projection Matrix** → projects data into a subspace
* **Least Squares** → finds the closest possible solution (best fit)

Together, they illustrate how linear algebra underpins many AI methods, especially in model fitting and dimensionality reduction.

---

## 📚 Requirements

See `requirements.txt`:

```
numpy==1.26.4
matplotlib==3.9.2
```

---

## 🏁 Author

**Your Name**
Math4AI – Linear Algebra Assignment 5
October 2025

```


