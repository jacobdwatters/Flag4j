package com.flag4j;


import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies operations which should be implemented by any matrix (rank 2 tensor).
 * @param <T> Matrix type.
 * @param <U> Dense Matrix type.
 * @param <V> Sparse Matrix type.
 * @param <W> Complex Matrix type.
 * @param <X> Matrix entry type.
 */
interface MatrixOperations<T, U, V, W, X> extends Operations<T, U, W, X> {


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


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public X fib(Matrix B);


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public X fib(SparseMatrix B);


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public CNumber fib(CMatrix B);


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    public CNumber fib(SparseCMatrix B);


    /**
     * Computes the direct sum of two matrices.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public T directSum(Matrix B);


    /**
     * Computes the direct sum of two matrices.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public V directSum(SparseMatrix B);


    /**
     * Computes the direct sum of two matrices.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public W directSum(CMatrix B);


    /**
     * Computes the direct sum of two matrices.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public SparseCMatrix directSum(SparseCMatrix B);


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public T invDirectSum(Matrix B);


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public V invDirectSum(SparseMatrix B);


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public W invDirectSum(CMatrix B);


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public SparseCMatrix invDirectSum(SparseCMatrix B);


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * an m-by-1 matrix.
     */
    public T sumCols();


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * an 1-by-n matrix.
     */
    public T sumRows();


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    public U addToEachCol(Vector b);


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    public T addToEachCol(SparseVector b);


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    public CMatrix addToEachCol(CVector b);


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    public SparseCMatrix addToEachCol(SparseCVector b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    public U addToEachRow(Vector b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    public U addToEachRow(SparseVector b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    public CMatrix addToEachRow(CVector b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    public SparseCMatrix addToEachRow(SparseCVector b);
}
