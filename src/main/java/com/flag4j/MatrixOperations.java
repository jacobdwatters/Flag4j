package com.flag4j;


/**
 * This interface specifies operations which should be implemented by any matrix (rank 2 tensor).
 * @param <T> Matrix type.
 * @param <U> Dense Matrix type.
 * @param <V> Sparse Matrix type.
 * @param <W> Complex Matrix type.
 * @param <X> Matrix entry type.
 */
interface MatrixOperations<T, U, V, W, X> extends Operations<T, U, W> {


    /**
     * Computes the element-wise addition between two matrices.
     * @param B Second matrix in the addition.
     * @return The result of adding the matrix B to this matrix element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    public U add(Matrix B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    public T add(SparseMatrix B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    public U add(CMatrix B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    public T add(SparseCMatrix B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    public U mult(Matrix B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    public U mult(SparseMatrix B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    public CMatrix mult(CMatrix B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    public W mult(SparseCMatrix B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public T elemMult(Matrix B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public V elemMult(SparseMatrix B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public W elemMult(CMatrix B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public SparseCMatrix elemMult(SparseCMatrix B);


    /**
     * Computes the element-wise division between two matrices.
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException If B contains any zero entries.
     */
    public T elemDiv(Matrix B);


    /**
     * Computes the element-wise division between two matrices.
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException If B contains any zero entries.
     */
    public V elemDiv(CMatrix B);


    /**
     * Computes the determinant of a square matrix.
     * @return The determinant of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    public X det();
}
