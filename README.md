[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=jacobdwatters_Flag4j&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=jacobdwatters_Flag4j)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=jacobdwatters_Flag4j&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=jacobdwatters_Flag4j)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=jacobdwatters_Flag4j&metric=coverage)](https://sonarcloud.io/summary/new_code?id=jacobdwatters_Flag4j)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=jacobdwatters_Flag4j&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=jacobdwatters_Flag4j)

# Flag4j
Flag4j is a fast linear algebra library for Java and provides operations and linear algebra routines for real/complex sparse/dense tensors, matrices, and vectors.

## State of Project as of 16-FEB-2024
Flag4j is currently in the final steps of development before an initial beta release. Nearly all features have been fully implemented or beta implementations.

Updates: 
- The QR, Hessenburg and Schur decompositions have had a complete overhall.
  - QR/Hessenburg/Schur: The way reflectors are computed and applied has been significantly improved in terms of performance, stability, and sensitivity to over(under)flow issues.
  - Schur: Some correcness issues have been ironed out and significant performance enhancements have been made. The decomposition uses an implicit double shifted QR algorithm.
    This means computing eigenvalues should be more accuract/correct and significantly faster in most cases.
- Tensor Solver: Added a solver which can solve tensor equations of the form $A \cdot X = B$ where $A, \ X$, and $B$ are tensors of arbirary rank and shape as long as the shapes are conducive to
  computing the tensor dot product `B = A.dot(X)`.
- Tensor Inverse: Added ability to compute the inverse of an arbitary invertible tensor $A$, $A^{-1}$ relative to the tensor product
  `X.tensorDot(Y, new int[]{n, n+1,..., X.rank()}, new int[]{0, 1,..., n})` where $n \in \mathbb{N}^+$.
___

## Features and Functionality
Highlighted features are a work in progress and currently not fully implemented.
### Algebraic Objects
- Complex Numbers
- Vectors
  - Real Dense Vector
  - Real Sparse Vector
  - Complex Dense Vector
  - Complex Sparse Vector
- Matrices
    - Real Dense Matrix
    - <mark>Real Sparse Matrix</mark>
    - Complex Dense Matrix
    - <mark>Complex Sparse Matrix</mark>
- Tensors
    - Real Dense Tensor
    - <mark>Real Sparse Tensor</mark>
    - Complex Dense Tensor
    - <mark>Complex Sparse Tensor</mark>

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
  - Full/Reduced
- Cholesky Decomposition
- Hessenburg Decomposition
- Eigen/Schur Decompositions
- Singular Value Decomposition

### Linear Solvers
- Exact solution for well determined systems
- Least Squares solution

### Linear and Homography Transformations
- Scale
- Shift
- <mark>Rotate</mark>
- <mark>Affine</mark>
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
