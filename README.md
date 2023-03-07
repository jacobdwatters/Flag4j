[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=jacobdwatters_JML&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=jacobdwatters_JML)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=jacobdwatters_JML&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=jacobdwatters_JML)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=jacobdwatters_JML&metric=coverage)](https://sonarcloud.io/summary/new_code?id=jacobdwatters_JML)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=jacobdwatters_JML&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=jacobdwatters_JML)

# Flag4j
Flag4j is a fast linear algebra library for Java and provides operations and linear algebra routines for real/complex sparse/dense tensors, matrices, and vectors.
Flag4j is currently in the initial steps of development.

___

## Features and Functionality
### Algebraic Objects
- Complex Numbers
- Vectors
  - Real Dense Vector
  - Real Sparse Vector
  - Complex Dense Vector
- Matrices
    - Real Dense Matrix
    - Real Sparse Matrix
    - Complex Dense Matrix
    - Complex Sparse Matrix
- Tensors
    - Real Dense Tensor
    - Real Sparse Tensor
    - Complex Dense Tensor
    - Complex Sparse Tensor

### Operations
- Basic Arithmetic Operations: Add, subtract, scalar/element multiply, scalar/element divide, etc.
- Basic Properties: Tensor/matrix/vector shape, non-zero entries, etc.
- Basic Manipulations: Insert, join/stack/augment, extract, etc.
- Basic Comparisons: Equal, same shape, etc.
- Vector Operations
  - Arithmetic: Inner/outer/cross product, vector norms, etc.
  - Comparisons: Parallel, orthogonal, etc.
- Matrix Operations
  - Arithmetic: Matrix multiplication, transpose, matrix norms, inverse, pseudo-inverse, etc.
  - Features: Matrix rank, symmetric, definiteness, eigenvalues and vectors, singularity, triangular, etc.
  - Comparisons: Similar, etc. 
- Tensor Operations:
  - Arithmetic: Tensor dot product, tensor transpose, etc.
  - Comparisons: Tensor rank/dimension, etc.

### Matrix Decompositions
- LU Decompositions
  - No pivoting
  - Partial Pivoting
  - Full Pivoting
- QR Decomposition
  - Householder Reflectors
- Cholesky Decomposition
- Eigen/Schur Decompositions
- Singular Value Decomposition

### Similarity Transformations
- Upper Hessenberg Form

### Linear Solvers
- Non-singular Linear System
- Least Squares

### Linear and Homogeneous Transformations
- Scale
- Shift
- Rotate
- Affine
- Projections
  - Orthographic
  - Perspective

### Random
- Random Complex Numbers
- Random Matrices
  - Orthogonal, Unitary, symmetric, triangular, etc.
- Random Vectors
- Random Tensors

### I/O
- Read/Write tensors, matrices, and vectors to/from a file.