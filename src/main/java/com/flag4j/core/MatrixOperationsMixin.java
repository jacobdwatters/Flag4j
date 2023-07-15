/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j.core;


import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.decompositions.SVD;


/**
 * This interface specifies operations which should be implemented by any matrix (rank 2 tensor).
 * @param <T> Matrix type.
 * @param <U> Dense Matrix type.
 * @param <V> Sparse Matrix type.
 * @param <W> Complex Matrix type.
 * @param <X> Matrix entry type.
 * @param <TT> Vector type equivalent.
 * @param <UU> Dense vector type.
 */
public interface MatrixOperationsMixin<
        T,
        U, V, W, X extends Number,
        TT extends VectorMixin<TT, UU, ?, ?, X, T, U, W>,
        UU extends VectorMixin<UU, UU, ?, CVector, X, U, U, CMatrix>> {

    /**
     * Computes the element-wise addition between two matrices.
     * @param B Second matrix in the addition.
     * @return The result of adding the matrix B to this matrix element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    U add(Matrix B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    T add(SparseMatrix B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    CMatrix add(CMatrix B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    W add(SparseCMatrix B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    U sub(Matrix B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    T sub(SparseMatrix B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    CMatrix sub(CMatrix B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    W sub(SparseCMatrix B);


    /**
     * Computes the element-wise addition of a matrix with a real sparse matrix. The result is stored in this matrix.
     * @param B The sparse matrix to add to this matrix.
     */
    void addEq(SparseMatrix B);


    /**
     * Computes the element-wise subtraction of this matrix with a real sparse matrix. The result is stored in this matrix.
     * @param B The sparse matrix to subtract from this matrix.
     */
    void subEq(SparseMatrix B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     * rows in matrix {@code B}.
     */
    U mult(Matrix B);


    /**
     * Computes the matrix-vector multiplication.
     * @param B Vector to multiply this matrix to.
     * @return The vector result from multiplying this matrix by the vector {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     * entries {@code B}.
     */
    UU mult(TT B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     * rows in matrix {@code B}.
     */
    U mult(T B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    U mult(SparseMatrix B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    CMatrix mult(CMatrix B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    CMatrix mult(SparseCMatrix B);


    /**
     * Computes matrix-vector multiplication.
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    UU mult(Vector b);


    /**
     * Computes matrix-vector multiplication.
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    UU mult(SparseVector b);


    /**
     * Computes matrix-vector multiplication.
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    CVector mult(CVector b);


    /**
     * Computes matrix-vector multiplication.
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    CVector mult(SparseCVector b);


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    U multTranspose(Matrix B);


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    U multTranspose(SparseMatrix B);


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {{@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    CMatrix multTranspose(CMatrix B);


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    CMatrix multTranspose(SparseCMatrix B);


    /**
     * Computes the matrix power with a given exponent. This is equivalent to multiplying a matrix to itself 'exponent'
     * times. Note, this method is preferred over repeated multiplication of a matrix as this method will be significantly
     * faster.
     * @param exponent The exponent in the matrix power.
     * @return The result of multiplying this matrix with itself 'exponent' times.
     */
    U pow(int exponent);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    T elemMult(Matrix B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    V elemMult(SparseMatrix B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    W elemMult(CMatrix B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    SparseCMatrix elemMult(SparseCMatrix B);


    /**
     * Computes the element-wise division between two matrices.
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException If B contains any zero entries.
     */
    T elemDiv(Matrix B);


    /**
     * Computes the element-wise division between two matrices.
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException If B contains any zero entries.
     */
    W elemDiv(CMatrix B);


    /**
     * Computes the determinant of a square matrix.
     * @return The determinant of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    X det();


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    X fib(Matrix B);


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    X fib(SparseMatrix B);


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    CNumber fib(CMatrix B);


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    CNumber fib(SparseCMatrix B);


    /**
     * Computes the direct sum of two matrices.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    T directSum(Matrix B);


    /**
     * Computes the direct sum of two matrices.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    T directSum(SparseMatrix B);


    /**
     * Computes the direct sum of two matrices.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    W directSum(CMatrix B);


    /**
     * Computes the direct sum of two matrices.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    W directSum(SparseCMatrix B);


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    T invDirectSum(Matrix B);


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    T invDirectSum(T B);


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    T invDirectSum(SparseMatrix B);


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    W invDirectSum(CMatrix B);


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    W invDirectSum(SparseCMatrix B);


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * an m-by-1 matrix.
     */
    T sumCols();


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * an 1-by-n matrix.
     */
    T sumRows();


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    U addToEachCol(Vector b);


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    T addToEachCol(SparseVector b);


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    CMatrix addToEachCol(CVector b);


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    W addToEachCol(SparseCVector b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    U addToEachRow(Vector b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    T addToEachRow(SparseVector b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    CMatrix addToEachRow(CVector b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    W addToEachRow(SparseCVector b);


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    U stack(Matrix B);


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    T stack(SparseMatrix B);


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    CMatrix stack(CMatrix B);


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    W stack(SparseCMatrix B);


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    U stack(Matrix B, int axis);


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    T stack(SparseMatrix B, int axis);


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    CMatrix stack(CMatrix B, int axis);


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    W stack(SparseCMatrix B, int axis);


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    U augment(Matrix B);


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    T augment(SparseMatrix B);


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    CMatrix augment(CMatrix B);


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    W augment(SparseCMatrix B);


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(Vector, int)} and {@link #augment(Vector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     * the vector b.
     */
    T stack(Vector b);


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(SparseVector, int)} and {@link #augment(SparseVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     * the vector b.
     */
    T stack(SparseVector b);


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CVector, int)} and {@link #augment(CVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     * the vector b.
     */
    W stack(CVector b);


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(SparseCVector, int)} and {@link #augment(SparseCVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     * the vector b.
     */
    W stack(SparseCVector b);


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Vector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Vector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    T stack(Vector b, int axis);


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(SparseVector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(SparseVector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    T stack(SparseVector b, int axis);


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(CVector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(CVector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    W stack(CVector b, int axis);


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(SparseCVector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(SparseCVector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    W stack(SparseCVector b, int axis);


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(Vector)} and {@link #stack(Vector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    T augment(Vector b);


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(SparseVector)} and {@link #stack(SparseVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    T augment(SparseVector b);


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CVector)} and {@link #stack(CVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    W augment(CVector b);


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(SparseCVector)} and {@link #stack(SparseCVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    W augment(SparseCVector b);


    /**
     * Get the row of this matrix at the specified index.
     * @param i Index of row to get.
     * @return The specified row of this matrix.
     */
    T getRow(int i);

    /**
     * Get the column of this matrix at the specified index.
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    T getCol(int j);


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     * @param colIdx Index of the column of this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd Ending row of the column (exclusive).
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     * @throws IllegalArgumentException If {@code rowStart} is less than 0.
     * @throws NegativeArraySizeException If {@code rowEnd} is less than {@code rowStart}.
     */
    TT getCol(int colIdx, int rowStart, int rowEnd);


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will be flattened then converted to a vector.
     * @return A vector equivalent to this matrix.
     */
    TT toVector();


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending row index of slice (exclusive).
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    T getSlice(int rowStart, int rowEnd, int colStart, int colEnd);


    /**
     * Get a specified column of this matrix at and below a specified row.
     *
     * @param rowStart Index of the row to begin at.
     * @param j Index of column to get.
     * @return The specified column of this matrix beginning at the specified row.
     * @throws NegativeArraySizeException If {@code rowStart} is larger than the number of rows in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code rowStart} or {@code j} is outside the bounds of this matrix.
     */
    T getColBelow(int rowStart, int j);


    /**
     * Get a specified row of this matrix at and after a specified column.
     *
     * @param colStart Index of the row to begin at.
     * @param i Index of the row to get.
     * @return The specified row of this matrix beginning at the specified column.
     * @throws NegativeArraySizeException If {@code colStart} is larger than the number of columns in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code i} or {@code colStart} is outside the bounds of this matrix.
     */
    T getRowAfter(int colStart, int i);


    /**
     * Sets a column of this matrix.
     * @param values Vector containing the new values for the matrix.
     * @param j Index of the column of this matrix to set.
     * @throws IllegalArgumentException If the number of entries in the {@code values} vector
     * is not the same as the number of rows in this matrix.
     * @throws IndexOutOfBoundsException If {@code j} is not within the bounds of this matrix.
     * @return A reference to this matrix.
     */
    T setCol(TT values, int j);


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #tr()}.
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    X trace();


    /**
     * Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     * Same as {@link #trace()}.
     * @return The trace of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    X tr();


    /**
     * Computes the inverse of this matrix.
     * @return The inverse of this matrix.
     */
    T inv();


    /**
     * Computes the pseudo-inverse of this matrix.
     * @return The pseudo-inverse of this matrix.
     */
    T pInv();


    /**
     * Computes the condition number of this matrix using {@link SVD SVD}.
     * Specifically, the condition number is computed as the maximum singular value divided by the minimum singular
     * value of this matrix.
     *
     * @return The condition number of this matrix (Assuming Frobenius norm).
     */
    double cond();

    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     * @return A vector containing the diagonal entries of this matrix.
     */
    TT getDiag();


    // This is specified here rather than in the ComplexMatrixMixin interface for compatibility purposes of real matrix
    // types in generic methods.
    /**
     * Compute the hermation transpose of this matrix. That is, the complex conjugate transpose of this matrix.
     * @return The complex conjugate transpose of this matrix.
     */
    T H();


    // This method is specified here for compatibility purposes in generic classes.
    /**
     * Copies this matrix.
     * @return A deep copy of this matrix.
     */
    T copy();
}
