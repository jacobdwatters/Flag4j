/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.core_old;


import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.*;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ErrorMessages;


/**
 * This interface specifies operations_old which should be implemented by any matrix (rank 2 tensor).
 * @param <T> MatrixOld type.
 * @param <U> Dense matrix type.
 * @param <V> Sparse matrix type.
 * @param <W> Complex matrix type.
 * @param <Y> Complex sparse matrix type.
 * @param <X> MatrixOld entry type.
 * @param <TT> VectorOld type equivalent.
 * @param <UU> Dense vector type.
 */
public interface MatrixOperationsMixin<
        T,
        U, V, W, Y, X extends Number,
        TT extends VectorMixin<TT, UU, ?, ?, ?, ?, ?, ?>,
        UU extends VectorMixin<UU, UU, ?, CVectorOld, X, U, U, CMatrixOld>> {

    /**
     * Computes the element-wise addition between two matrices.
     * @param B Second matrix in the addition.
     * @return The result of adding the matrix B to this matrix element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    U add(MatrixOld B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    T add(CooMatrixOld B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    T add(CsrMatrixOld B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    W add(CsrCMatrixOld B);



    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    CMatrixOld add(CMatrixOld B);


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    W add(CooCMatrixOld B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    U sub(MatrixOld B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    T sub(CooMatrixOld B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    T sub(CsrMatrixOld B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    W sub(CsrCMatrixOld B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    CMatrixOld sub(CMatrixOld B);


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    W sub(CooCMatrixOld B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix {@code B}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     * rows in matrix {@code B}.
     */
    U mult(MatrixOld B);


    /**
     * Computes the matrix-vector multiplication.
     * @param b VectorOld to multiply this matrix to.
     * @return The vector result from multiplying this matrix by the vector {@code b}.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of
     * entries {@code b}.
     */
    UU mult(TT b);


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
    U mult(CooMatrixOld B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    U mult(CsrMatrixOld B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    CMatrixOld mult(CMatrixOld B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    CMatrixOld mult(CooCMatrixOld B);


    /**
     * Computes the matrix multiplication between two matrices.
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    CMatrixOld mult(CsrCMatrixOld B);


    /**
     * Computes matrix-vector multiplication.
     * @param b VectorOld in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    UU mult(VectorOld b);


    /**
     * Computes matrix-vector multiplication.
     * @param b VectorOld in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    UU mult(CooVectorOld b);


    /**
     * Computes matrix-vector multiplication.
     * @param b VectorOld in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    CVectorOld mult(CVectorOld b);


    /**
     * Computes matrix-vector multiplication.
     * @param b VectorOld in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    CVectorOld mult(CooCVectorOld b);


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    default U multTranspose(MatrixOld B) {
        return this.mult(B.T());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    default U multTranspose(CooMatrixOld B) {
        return this.mult(B.T());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    default U multTranspose(CsrMatrixOld B) {
        return this.mult(B.T());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {{@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    default CMatrixOld multTranspose(CMatrixOld B) {
        return this.mult(B.T());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    default CMatrixOld multTranspose(CooCMatrixOld B) {
        return this.mult(B.T());
    }


    /**
     * Multiplies this matrix with the transpose of the {@code B} tensor as if by
     * {@code this.mult(B.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(B.T())}.
     * @param B The second matrix in the multiplication and the matrix to transpose/
     * @return The result of multiplying this matrix with the transpose of {@code B}.
     */
    default CMatrixOld multTranspose(CsrCMatrixOld B) {
        return this.mult(B.T());
    }


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
    T elemMult(MatrixOld B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    V elemMult(CooMatrixOld B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    W elemMult(CMatrixOld B);


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    Y elemMult(CooCMatrixOld B);


    /**
     * Computes the element-wise division between two matrices.
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException If B contains any zero entries.
     */
    T elemDiv(MatrixOld B);


    /**
     * Computes the element-wise division between two matrices.
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException If B contains any zero entries.
     */
    W elemDiv(CMatrixOld B);


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
    X fib(MatrixOld B);


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    X fib(CooMatrixOld B);


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    CNumber fib(CMatrixOld B);


    /**
     * Computes the Frobenius inner product of two matrices.
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    CNumber fib(CooCMatrixOld B);


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * a vectors of length m.
     */
    UU sumCols();


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     *
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * a vector of length n.
     */
    UU sumRows();


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b VectorOld to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    U addToEachCol(VectorOld b);


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b VectorOld to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    U addToEachCol(CooVectorOld b);


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b VectorOld to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    CMatrixOld addToEachCol(CVectorOld b);


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     * @param b VectorOld to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    CMatrixOld addToEachCol(CooCVectorOld b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b VectorOld to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    U addToEachRow(VectorOld b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b VectorOld to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    U addToEachRow(CooVectorOld b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b VectorOld to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    CMatrixOld addToEachRow(CVectorOld b);


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     * @param b VectorOld to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    CMatrixOld addToEachRow(CooCVectorOld b);


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(MatrixOld, int)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    U stack(MatrixOld B);


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(MatrixOld, int)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    T stack(CooMatrixOld B);


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(MatrixOld, int)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    default T stack(CsrMatrixOld B) {
        return this.stack(B.toCoo());
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(MatrixOld, int)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    default W stack(CsrCMatrixOld B) {
        return this.stack(B.toCoo());
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(MatrixOld, int)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    CMatrixOld stack(CMatrixOld B);


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(MatrixOld, int)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    W stack(CooCMatrixOld B);


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(MatrixOld)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(MatrixOld)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default U stack(MatrixOld B, int axis){
        U stacked;

        if(axis==0) {
            stacked = this.augment(B);
        } else if(axis==1) {
            stacked = this.stack(B);
        } else {
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));
        }

        return stacked;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(MatrixOld)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(MatrixOld)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default T stack(CooMatrixOld B, int axis){
        T stacked;

        if(axis==0) {
            stacked = this.augment(B);
        } else if(axis==1) {
            stacked = this.stack(B);
        } else {
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));
        }

        return stacked;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(MatrixOld)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(MatrixOld)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default W stack(CsrCMatrixOld B, int axis){
        return stack(B.toCoo(), axis);
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(MatrixOld)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(MatrixOld)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default T stack(CsrMatrixOld B, int axis){
        return stack(B.toCoo(), axis);
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(MatrixOld)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(MatrixOld)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default CMatrixOld stack(CMatrixOld B, int axis){
        CMatrixOld stacked;

        if(axis==0) {
            stacked = this.augment(B);
        } else if(axis==1) {
            stacked = this.stack(B);
        } else {
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));
        }

        return stacked;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #augment(MatrixOld)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(MatrixOld)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(MatrixOld)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default W stack(CooCMatrixOld B, int axis){
        W stacked;

        if(axis==0) {
            stacked = this.augment(B);
        } else if(axis==1) {
            stacked = this.stack(B);
        } else {
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));
        }

        return stacked;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #stack(MatrixOld, int)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    U augment(MatrixOld B);


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #stack(MatrixOld, int)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    T augment(CooMatrixOld B);


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #stack(MatrixOld, int)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    default T augment(CsrMatrixOld B) {
        return augment(B.toCoo());
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #stack(MatrixOld, int)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    default W augment(CsrCMatrixOld B) {
        return augment(B.toCoo());
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #stack(MatrixOld, int)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    CMatrixOld augment(CMatrixOld B);


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(MatrixOld)} and {@link #stack(MatrixOld, int)}.
     *
     * @param B MatrixOld to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    W augment(CooCMatrixOld B);


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(VectorOld, int)} and {@link #augment(VectorOld)}.
     *
     * @param b VectorOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     * the vector b.
     */
    T stack(VectorOld b);


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CooVectorOld, int)} and {@link #augment(CooVectorOld)}.
     *
     * @param b VectorOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     * the vector b.
     */
    T stack(CooVectorOld b);


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CVectorOld, int)} and {@link #augment(CVectorOld)}.
     *
     * @param b VectorOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     * the vector b.
     */
    W stack(CVectorOld b);


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CooCVectorOld, int)} and {@link #augment(CooCVectorOld)}.
     *
     * @param b VectorOld to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     * the vector b.
     */
    W stack(CooCVectorOld b);


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b VectorOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(VectorOld)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(VectorOld)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default T stack(VectorOld b, int axis) {
        T stacked;

        if(axis==0) stacked = this.augment(b);
        else if(axis==1) stacked = this.stack(b);
        else throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));

        return stacked;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b VectorOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(CooVectorOld)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(CooVectorOld)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default T stack(CooVectorOld b, int axis) {
        T stacked;

        if(axis==0) stacked = this.augment(b);
        else if(axis==1) stacked = this.stack(b);
        else throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));

        return stacked;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b VectorOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(CVectorOld)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(CVectorOld)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default W stack(CVectorOld b, int axis) {
        W stacked;

        if(axis==0) stacked = this.augment(b);
        else if(axis==1) stacked = this.stack(b);
        else throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));

        return stacked;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b VectorOld to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(CooCVectorOld)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(CooCVectorOld)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    default W stack(CooCVectorOld b, int axis) {
        W stacked;

        if(axis==0) stacked = this.augment(b);
        else if(axis==1) stacked = this.stack(b);
        else throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));

        return stacked;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(VectorOld)} and {@link #stack(VectorOld, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    T augment(VectorOld b);


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CooVectorOld)} and {@link #stack(CooVectorOld, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    T augment(CooVectorOld b);


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CVectorOld)} and {@link #stack(CVectorOld, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    W augment(CVectorOld b);


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CooCVectorOld)} and {@link #stack(CooCVectorOld, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    W augment(CooCVectorOld b);


    /**
     * Get the row of this matrix at the specified index.
     * @param i Index of row to get.
     * @return The specified row of this matrix.
     */
    TT getRow(int i);


    /**
     * Get the column of this matrix at the specified index.
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    TT getCol(int j);


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
    TT getColBelow(int rowStart, int j);


    /**
     * Get a specified row of this matrix at and after a specified column.
     *
     * @param colStart Index of the row to begin at.
     * @param i Index of the row to get.
     * @return The specified row of this matrix beginning at the specified column.
     * @throws NegativeArraySizeException If {@code colStart} is larger than the number of columns in this matrix.
     * @throws ArrayIndexOutOfBoundsException If {@code i} or {@code colStart} is outside the bounds of this matrix.
     */
    TT getRowAfter(int colStart, int i);


    /**
     * Sets a column of this matrix.
     * @param values VectorOld containing the new values for the matrix.
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
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     * @return A vector containing the diagonal entries of this matrix.
     */
    TT getDiag();


    // This is specified here rather than in the ComplexMatrixMixin interface for compatibility purposes of real matrix
    // types in generic methods.
    /**
     * Compute the hermitian transpose of this matrix. That is, the complex conjugate transpose of this matrix.
     * @return The complex conjugate transpose of this matrix.
     */
    T H();


    // This method is specified here in addition to the tensor mixin for compatibility purposes in generic classes.
    /**
     * Copies this matrix.
     * @return A deep copy of this matrix.
     */
    T copy();


    // This method is specified here in addition to the tensor mixin for compatibility purposes in generic classes.
    /**
     * Computes scalar multiplication of a matrix.
     * @param factor Scalar value to multiply with matrix.
     * @return The result of multiplying this matrix by the specified scalar.
     */
    T mult(double factor);
}
