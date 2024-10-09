/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.arrays.backend;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.operations.TransposeDispatcher;
import org.flag4j.linalg.operations.common.field_ops.CompareField;
import org.flag4j.linalg.operations.dense.field_ops.DenseFieldDeterminant;
import org.flag4j.linalg.operations.dense.field_ops.DenseFieldMatMultDispatcher;
import org.flag4j.linalg.operations.dense.field_ops.DenseFieldProperties;
import org.flag4j.linalg.operations.dense.field_ops.DenseFieldTensorDot;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;


/**
 * <p>The base class for all dense {@link Field} matrices.</p>
 * <p>The {@link #entries} of a dense field matrix are mutable but the {@link #shape} is fixed.</p>
 *
 * @param <T> The type of this dense field matrix.
 * @param <U> Type of sparse COO matrix equivalent to {@code T}. This type parameter is required because some operations (e.g.
 * {@link #toCoo()}) may result in a sparse COO matrix.
 * @param <V> Type of sparse CSR matrix equivalent to {@code T}. This type parameter is required because some operations (e.g.
 * {@link #toCsr()}) may result in a sparse CSR matrix.
 * @param <W> The type of the dense vector similar to {@code T}.
 * @param <Y> The type (or wrapper of) an individual element of the matrix.
 */
public abstract class DenseFieldMatrixBase<T extends DenseFieldMatrixBase<T, U, V, W, Y>,
        U extends CooFieldMatrixBase<U, T, ?, W, Y>,
        V extends CsrFieldMatrixBase<V, T, ?, W, Y>,
        W extends DenseFieldVectorBase<W, T, ?, Y>,
        Y extends Field<Y>>
        extends FieldTensorBase<T, T, Y>
        implements DenseMatrixMixin<T, U, W, Y> {

    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected DenseFieldMatrixBase(Shape shape, Field<Y>[] entries) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);
        ValidateParameters.ensureEquals(entries.length, shape.totalEntriesIntValueExact());
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     *
     * @param b TensorOld to compare this tensor to.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     *
     * @return True if the argument {@code tensor} is the same shape as this tensor and all entries are 'close', i.e.
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively satisfy
     * {@code |x-y| <= (atol + rtol*|y|)}. Otherwise, returns false.
     *
     * @see #allClose(DenseTensorMixin)
     */
    @Override
    public boolean allClose(T b, double relTol, double absTol) {
        if(!shape.equals(b.shape)) return false;
        return CompareField.allClose(entries, b.entries, relTol, absTol);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public abstract T makeLikeTensor(Shape shape, Field<Y>[] entries);


    /**
     * Constructs a matrix of the same type as this matrix with the given the shape filled with the specified fill value.
     *
     * @param shape Shape of the matrix to construct.
     * @param fillValue Value to fill this matrix with.
     *
     * @return A matrix of the same type as this tensor with the given the shape and entries.
     */
    public abstract T makeLikeTensor(Shape shape, Y fillValue);


    /**
     * Constructs a vector of similar type to this matrix with the given {@code entries}.
     * @param entries Entries of the vector.
     * @return A vector of similar type to this matrix with the given {@code entries}.
     */
    public abstract W makeLikeVector(Field<Y>... entries);


    /**
     * Constructs a sparse CSR matrix of similar type to this dense matrix.
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero entries of the CSR matrix.
     * @param rowPointers Row pointers of the CSR matrix.
     * @param colIndices Column indices of the non-zero entries in the CSR matrix.
     * @return A sparse CSR matrix with the specified shape and non-zero entries.
     */
    public abstract V makeLikeCsrMatrix(Shape shape, Field<Y>[] entries, int[] rowPointers, int[] colIndices);


    /**
     * Constructs a sparse COO matrix of similar type to this dense matrix.
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero entries of the COO matrix.
     * @param rowIndices Row indices of the non-zero entries in the COO matrix.
     * @param colIndices Column indices of the non-zero entries in the COO matrix.
     * @return A sparse COO matrix with the specified shape and non-zero entries.
     */
    public abstract U makeLikeCooMatrix(Shape shape, Field<Y>[] entries, int[] rowIndices, int[] colIndices);
    

    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 TensorOld to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public T tensorDot(T src2, int[] aAxes, int[] bAxes) {
        return DenseFieldTensorDot.tensorDot(this, src2, aAxes, bAxes);
    }


    /**
     * Computes the tensor dot product of this tensor with a second tensor. That is, sums the product of two tensor
     * elements over the last axis of this tensor and the second-to-last axis of {@code src2}. If both tensors are
     * rank 2, this is equivalent to matrix multiplication.
     *
     * @param src2 TensorOld to compute dot product with this tensor.
     *
     * @return The tensor dot product over the last axis of this tensor and the second to last axis of {@code src2}.
     *
     * @throws IllegalArgumentException If this tensors shape along the last axis does not match {@code src2} shape
     *                                  along the second-to-last axis.
     */
    @Override
    public T tensorDot(T src2) {
        return DenseFieldTensorDot.tensorDot(this, src2);
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public T H(int axis1, int axis2) {
        ValidateParameters.ensureValidIndices(2, axis1, axis2);
        return (T) TransposeDispatcher.dispatchHermitian(this);
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public T H(int... axes) {
        return (T) TransposeDispatcher.dispatchHermitian(this);
    }


    /**
     * Constructs the zero matrix with specified shape and zero value defined with respect to a field element.
     * @param rows The number of rows in this matrix.
     * @param rows The number of columns in this matrix.
     * @param value Elements of field to get zero value from.
     * @return The zero matrix with specified shape and zero value defined with respect to a field element.
     */
    public T getZeroMatrix(int rows, int cols, Y fieldValue) {
        return makeLikeTensor(new Shape(rows, cols), fieldValue.getZero());
    }


    /**
     * Gets the number of rows in this matrix.
     *
     * @return The number of rows in this matrix.
     */
    @Override
    public int numRows() {
        return numRows;
    }


    /**
     * Gets the number of columns in this matrix.
     *
     * @return The number of columns in this matrix.
     */
    @Override
    public int numCols() {
        return numCols;
    }


    /**
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.</p>
     *
     * <p>Same as {@link #trace()}.</p>
     *
     * @return The trace of this matrix.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Y tr() {
        ValidateParameters.ensureSquareMatrix(this.shape);
        Field<Y> sum = entries[0];
        int colsOffset = this.numCols+1;

        for(int i=1; i<this.numRows; i++) {
            sum = sum.add((Y) this.entries[i*colsOffset]);
        }

        return (Y) sum;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     *
     * @see #isTri()
     * @see #isTriL()
     * @see #isDiag()
     */
    @Override
    public boolean isTriU() {
        if(!isSquare()) return false;

        // Ensure lower half is zeros.
        for(int i=1; i<numRows; i++) {
            for(int j=0; j<i; j++) {
                if(!entries[i*numCols + j].equals(1)) {
                    return false; // No need to continue.
                }
            }
        }

        return true;
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     *
     * @see #isTri()
     * @see #isTriU()
     * @see #isDiag()
     */
    @Override
    public boolean isTriL() {
        if(!isSquare()) return false;

        // Ensure upper half is zeros.
        for(int i=0; i<numRows; i++) {
            for(int j=i+1; j<numCols; j++) {
                if(!entries[i*numCols + j].equals(1)) {
                    return false; // No need to continue.
                }
            }
        }

        return true;
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.
     *
     * @return True if this matrix is singular or non-square. Otherwise, returns false.
     *
     * @see #isInvertible()
     */
    @Override
    public boolean isSingular() {
        return isSquare() || det().mag() < Flag4jConstants.EPS_F64;
    }


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     *
     * @see #isCloseToI()
     */
    @Override
    public boolean isI() {
        int pos = 0;

        if(isSquare()) {
            for(int i=0; i<numRows; i++) {
                for(int j=0; j<numCols; j++) {
                    if((i==j && !entries[pos].isOne()) || i!=j && !entries[pos].isZero())
                        return false; // No need to continue

                    pos++;
                }
            }
        } else {
            return false; // An identity matrix must be square.
        }

        // If we make it to this point this matrix must be an identity matrix.
        return true;
    }


    /**
     * Checks that this matrix is close to the identity matrix.
     *
     * @return True if this matrix is approximately the identity matrix.
     *
     * @see #isI()
     */
    @Override
    public boolean isCloseToI() {
        return DenseFieldProperties.isCloseToIdentity(this);
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     *
     * @throws LinearAlgebraException If this matrix is not square.
     */
    @Override
    public Y det() {
        return DenseFieldDeterminant.det(this);
    }


    /**
     * <p>Computes the rank of this matrix (i.e. the number of linearly independent rows/columns in this matrix).</p>
     *
     * <p>Note that here, rank is <b>NOT</b> the same as a tensor rank (i.e. number of indices needed to specify an entry in
     * the tensor).</p>
     *
     * @return The matrix rank of this matrix.
     */
    @Override
    public int matrixRank() {
        // TODO: Implementation.
        //  (This does not make sense to have implemented here as it requires the SVDOld. Need to implement in
        //  the real and complex-valued matrix only and not the general field matrix).
        return 0;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number of rows in matrix {@code b}.
     */
    @Override
    public T mult(T b) {
        return makeLikeTensor(new Shape(numRows, b.numCols), DenseFieldMatMultDispatcher.dispatch(this, b));
    }


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose.
     *
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    @Override
    public T multTranspose(T b) {
        return makeLikeTensor(new Shape(numRows, b.numCols), DenseFieldMatMultDispatcher.dispatchTranspose(this, b));
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param b Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and b have different shapes.
     */
    @Override
    public Y fib(T b) {
        ValidateParameters.ensureEqualShape(this.shape, b.shape);
        return this.H().mult(b).trace();
    }


    /**
     * The transpose of this matrix.
     * @return The transpose of this matrix.
     */
    @Override
    public T T() {
        return (T) TransposeDispatcher.dispatch(this);
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public T T(int axis1, int axis2) {
        ValidateParameters.ensureValidIndices(2, axis1, axis2);
        if(axis1==axis2) return copy();

        return (T) TransposeDispatcher.dispatch(this);
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public T T(int... axes) {
        ValidateParameters.ensureArrayLengthsEq(2, axes.length);
        ValidateParameters.ensureValidIndices(2, axes[0], axes[1]);

        return (T) TransposeDispatcher.dispatch(this);
    }


    /**
     * Stacks matrices along columns. <br>
     *
     * @param b MatrixOld to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(TensorBase, int)
     * @see #augment(FieldMatrix)
     */
    @Override
    public T stack(T b) {
        ValidateParameters.ensureArrayLengthsEq(this.numCols, b.numCols);
        Shape stackedShape = new Shape(this.numRows + b.numRows, this.numCols);
        Field<Y>[] stackedEntries = new Field[stackedShape.totalEntries().intValueExact()];

        System.arraycopy(this.entries, 0, stackedEntries, 0, this.entries.length);
        System.arraycopy(b.entries, 0, stackedEntries, this.entries.length, b.entries.length);

        return makeLikeTensor(stackedShape, stackedEntries);
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b MatrixOld to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(DenseFieldMatrixBase) 
     * @see #stack(MatrixMixin, int)
     */
    @Override
    public T augment(T b) {
        ValidateParameters.ensureArrayLengthsEq(numRows, b.numRows);

        int augNumCols = numCols + b.numCols;
        Shape augShape = new Shape(numRows, augNumCols);
        Field<Y>[] augEntries = new Field[numRows*augNumCols];

        // Copy entries from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(entries, i*numCols, augEntries, i*augNumCols, numCols);

            int augOffset = i*augNumCols + numCols;
            int bOffset = i*b.numCols;
            for(int j=0; j<b.numCols; j++) {
                augEntries[augOffset + j] = b.entries[bOffset + j];
            }
        }

        return makeLikeTensor(augShape, augEntries);
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    @Override
    public T augment(W b) {
        ValidateParameters.ensureArrayLengthsEq(numRows, b.size);
        Field<Y>[] augmented = new Field[numRows*(numCols + 1)];

        // Copy entries from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(entries, i*numCols, augmented, i*(numCols+1), numCols);
            augmented[i*(numCols+1) + numCols] = b.entries[i];
        }

        return makeLikeTensor(new Shape(numRows, numCols+1), augmented);
    }


    /**
     * Swaps specified rows in the matrix. This is done in place.
     *
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public T swapRows(int rowIndex1, int rowIndex2) {
        ValidateParameters.ensureValidIndices(numRows, rowIndex1, rowIndex2);

        int row1Offset = rowIndex1*numCols;
        int row2Offset = rowIndex2*numCols;

        if(rowIndex1 != rowIndex2) {
            Field<Y> temp;

            for(int j=0; j<numCols; j++) {
                // Swap elements.
                temp = entries[row1Offset + j];
                entries[row1Offset + j] = entries[row2Offset + j];
                entries[row2Offset + j] = temp;
            }
        }

        return (T) this;
    }


    /**
     * Swaps specified columns in the matrix. This is done in place.
     *
     * @param colIndex1 Index of the first column to swap.
     * @param colIndex2 Index of the second column to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public T swapCols(int colIndex1, int colIndex2) {
        ValidateParameters.ensureValidIndices(numCols, colIndex1, colIndex2);

        if(colIndex1 != colIndex2) {
            Field<Y> temp;

            for(int i=0; i<numRows; i++) {
                // Swap elements.
                int idx = i*numCols;
                ArrayUtils.swap(entries, idx + colIndex1, idx + colIndex2);
            }
        }

        return (T) this;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     *
     * @see #isAntiSymmetric()
     */
    @Override
    public boolean isSymmetric() {
        if(this==null) return false;
        if(this.entries.length==0) return true;

        return numRows==numCols && this.equals(this.T());
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        if(this==null) return false;
        if(this.entries.length==0) return true;

        return numRows==numCols && this.equals(this.H());
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     *
     * @see #isSymmetric()
     */
    @Override
    public boolean isAntiSymmetric() {
        if(this==null) return false;
        if(this.entries.length==0) return true;

        return numRows==numCols && this.equals(this.T().mult(entries[0].getOne().addInv()));
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        return numRows == numCols && DenseFieldProperties.isCloseToIdentity(this.mult(this.T()));
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     *
     * @return A copy of this matrix with the specified row removed.
     */
    @Override
    public T removeRow(int rowIndex) {
        Shape copyShape = new Shape(numRows-1, numCols);
        Field<Y>[] copyEntries = new Field[(numRows-1)*numCols];

        int row = 0;

        for(int i=0; i<this.numRows; i++) {
            if(i!=rowIndex) {
                System.arraycopy(this.entries, i*numCols, copyEntries, row*numCols, numCols);
                row++;
            }
        }

        return makeLikeTensor(new Shape(numRows-1, numCols), copyEntries);
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public T removeRows(int... rowIndices) {
        Shape copyShape = new Shape(numRows-rowIndices.length, numCols);
        Field<Y>[] copyEntries = new Field[(numRows-rowIndices.length)*numCols];

        int row = 0;

        for(int i=0; i<this.numRows; i++) {
            if(ArrayUtils.notContains(rowIndices, i)) {
                System.arraycopy(this.entries, i*numCols, copyEntries, row*numCols, numCols);
                row++;
            }
        }

        return makeLikeTensor(new Shape(numRows-rowIndices.length, numCols), copyEntries);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCol(int colIndex) {
        int copyNumCols = numCols-1;
        Shape copyShape = new Shape(numRows, copyNumCols);
        Field<Y>[] copyEntries = new Field[numRows*copyNumCols];

        int col;

        for(int i=0; i<this.numRows; i++) {
            int rowOffset = i*numCols;
            int copyOffset = i*copyNumCols;
            col = 0;

            for(int j=0; j<this.numCols; j++) {
                if(j!=colIndex) {
                    copyEntries[copyOffset + col] = this.entries[rowOffset + j];
                    col++;
                }
            }
        }

        return makeLikeTensor(new Shape(numRows, numCols-1), copyEntries);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCols(int... colIndices) {
        int copyNumCols = this.numCols-colIndices.length;
        Shape copyShape = new Shape(numRows, copyNumCols);
        Field<Y>[] copyEntries = new Field[numRows*copyNumCols];

        int col;

        for(int i=0; i<this.numRows; i++) {
            int rowOffset = i*numCols;
            int copyOffset = i*copyNumCols;
            col = 0;

            for(int j=0; j<this.numCols; j++) {
                if(ArrayUtils.notContains(colIndices, j)) {
                    copyEntries[copyOffset + col] = this.entries[rowOffset + j];
                    col++;
                }
            }
        }

        return makeLikeTensor(new Shape(numRows, numCols-colIndices.length), copyEntries);
    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set within this matrix.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     *                                  fit completely within this matrix.
     */
    @Override
    public T setSlice(T values, int rowStart, int colStart) {
        ValidateParameters.ensureValidIndices(numRows, rowStart);
        ValidateParameters.ensureValidIndices(numCols, colStart);

        for(int i=0; i<values.numRows; i++) {
            int src1Offset = (i+rowStart)*numCols + colStart;
            int src2RowOffset = i*values.numCols;

            for(int j=0; j<values.numCols; j++) {
                this.entries[src1Offset + j] = values.entries[src2RowOffset + j];
            }
        }

        return (T) this;
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A copy of this matrix with the given slice set to the specified values.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public T setSliceCopy(T values, int rowStart, int colStart) {
        return copy().setSlice(values, rowStart, colStart);
    }


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending row index of slice (exclusive).
     *
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException       If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    @Override
    public T getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        ValidateParameters.ensureValidIndices(numRows, rowStart, rowEnd);
        ValidateParameters.ensureValidIndices(numCols, colStart, colEnd);

        int sliceRows = rowEnd-rowStart;
        int sliceCols = colEnd-colStart;
        int destPos = 0;
        int srcPos;
        int end;
        Field<Y>[] slice = new Field[sliceRows*sliceCols];

        for(int i=rowStart; i<rowEnd; i++) {
            srcPos = i*numCols + colStart;
            end = srcPos + colEnd - colStart;

            while(srcPos < end) {
                slice[destPos++] = entries[srcPos++];
            }
        }

        return makeLikeTensor(new Shape(sliceRows, sliceCols), slice);
    }


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row Row index to set.
     * @param col Column index to set.
     *
     * @return A reference to this matrix.
     */
    @Override
    public T set(Y value, int row, int col) {
        entries[row*numCols + col] = value;
        return (T) this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public T setValues(Y[][] values) {
        ValidateParameters.ensureEquals(numRows, values.length);
        ValidateParameters.ensureEquals(numCols, values[0].length);

        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j<numCols; j++)
                entries[rowOffset + j] = values[i][j];
        }

        return (T) this;
    }


    /**
     * Extracts the upper-triangular portion of this matrix with a specified diagonal offset. All other entries of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for upper-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all entries at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all entries at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all entries at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The upper-triangular portion of this matrix with a specified diagonal offset. All other entries of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public T getTriU(int diagOffset) {
        ValidateParameters.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        T result = makeLikeTensor(shape, entries[0].getZero());

        // Extract the upper triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=Math.max(0, i + diagOffset); j<numCols; j++) {
                if (j >= i + diagOffset) {
                    result.entries[rowOffset + j] = entries[rowOffset + j];
                }
            }
        }

        return result;
    }


    /**
     * Extracts the lower-triangular portion of this matrix with a specified diagonal offset. All other entries of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for lower-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all entries at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all entries at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all entries at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The lower-triangular portion of this matrix with a specified diagonal offset. All other entries of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public T getTriL(int diagOffset) {
        ValidateParameters.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        T result = makeLikeTensor(shape, entries[0].getZero());

        // Extract the lower triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j <= Math.min(numCols - 1, i + diagOffset); j++) {
                if(j <= i + diagOffset) {
                    result.entries[rowOffset + j] = this.entries[rowOffset + j];
                }
            }
        }

        return result;
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link T}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        T src2 = (T) object;

        return shape.equals(src2.shape) && Arrays.equals(entries, src2.entries);
    }


    /**
     * Computes the element-wise multiplication of two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public void elemMultEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i]  = entries[i].mult((Y) b.entries[i]);
    }


    /**
     * Computes the element-wise sum between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public void addEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].add((Y) b.entries[i]);
    }


    /**
     * Computes the element-wise difference between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public void subEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].sub((Y) b.entries[i]);
    }


    /**
     * Computes the element-wise division between two tensors and stores the result in this tensor.
     *
     * @param b The denominator tensor in the element-wise quotient.
     *
     * @throws TensorShapeException If this tensor and {@code b}'s shape are not equal.
     */
    @Override
    public void divEq(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);

        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].div((Y) b.entries[i]);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param b The denominator tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor and {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b}'s shape are not equal.
     */
    @Override
    public T div(T b) {
        ValidateParameters.ensureEqualShape(shape, b.shape);
        Field<Y>[] quotient = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            quotient[i] = entries[i].div((Y) b.entries[i]);

        return makeLikeTensor(shape, quotient);
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the
     *                                  number of entries in the vector {@code b}.
     */
    @Override
    public W mult(W b) {
        Field<Y>[] dest = DenseFieldMatMultDispatcher.dispatch(this, b);
        return makeLikeVector(dest);
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will first be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public W toVector() {
        return makeLikeVector(entries.clone());
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param rowIdx Index of row to get.
     *
     * @return The specified row of this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code rowIdx} is less than zero or greater than/equal to
     *                                        the number of rows in this matrix.
     */
    @Override
    public W getRow(int rowIdx) {
        ValidateParameters.ensureIndexInBounds(numRows, rowIdx);
        int start = rowIdx*numCols;
        int stop = start+numCols;

        return makeLikeVector(Arrays.copyOfRange(this.entries, start, stop));
    }


    /**
     * Gets a specified row of this matrix between {@code colStart} (inclusive) and {@code colEnd} (exclusive).
     *
     * @param rowIdx Index of the row of this matrix to get.
     * @param colStart Starting column of the row (inclusive).
     * @param colEnd Ending column of the row (exclusive).
     *
     * @return The row at index {@code rowIdx} of this matrix between the {@code colStart} and {@code colEnd}
     * indices.
     *
     * @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException  If {@code colEnd} is less than {@code colStart}.
     */
    @Override
    public W getRow(int rowIdx, int colStart, int colEnd) {
        ValidateParameters.ensureIndexInBounds(numCols, colStart, colEnd);
        ValidateParameters.ensureGreaterEq(colStart, colEnd);
        int start = rowIdx*numCols+colStart;
        int stop = start+colEnd;

        return makeLikeVector(Arrays.copyOfRange(this.entries, start, stop));
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param colIdx Index of column to get.
     *
     * @return The specified column of this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If {@code colIdx} is less than zero or greater than/equal to
     *                                        the number of columns in this matrix.
     */
    @Override
    public W getCol(int colIdx) {
        ValidateParameters.ensureValidIndices(numCols, colIdx);
        Field<Y>[] col = new Field[numRows];

        for(int i=0; i<numRows; i++)
            col[i] = entries[i*numCols + colIdx];

        return makeLikeVector(col);
    }


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     *
     * @param colIdx Index of the column of this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd Ending row of the column (exclusive).
     *
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     *
     * @throws @throws                  IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the
     *                                  shape of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is less than {@code rowStart}.
     */
    @Override
    public W getCol(int colIdx, int rowStart, int rowEnd) {
        ValidateParameters.ensureValidIndices(numRows, rowStart, rowEnd);
        ValidateParameters.ensureGreaterEq(rowEnd, rowStart);
        Field<Y>[] col = new Field[numRows];

        for(int i=rowStart; i<rowEnd; i++)
            col[i] = entries[i*numCols + colIdx];

        return makeLikeVector(col);
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    @Override
    public W getDiag() {
        int newSize = Math.min(numRows, numCols);
        Field<Y>[] diag = new Field[newSize];

        int idx = 0;
        for(int i=0; i<newSize; i++) {
            diag[i] = this.entries[idx];
            idx += numCols + 1;
        }

        return makeLikeVector(diag);
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows of this matrix.
     */
    @Override
    public T setCol(W values, int colIndex) {
        ValidateParameters.ensureArrayLengthsEq(values.size, this.numRows);

        int rowOffset = 0;
        for(int i=0; i<values.size; i++) {
            entries[rowOffset + colIndex] = values.entries[i];
            rowOffset += numCols;
        }

        return (T) this;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IndexOutOfBoundsException If the values vector has a different length than the number of rows of this matrix.
     */
    @Override
    public T setRow(W values, int rowIndex) {
        ValidateParameters.ensureArrayLengthsEq(values.size, this.numCols);
        int rowOffset = rowIndex*numCols;

        for(int i=0; i<values.size; i++)
            super.entries[rowOffset + i] = values.entries[i];

        return (T) this;
    }


    /**
     * Computes the element-wise absolute value of this matrix.
     *
     * @return The element-wise absolute value of this matrix.
     */
    @Override
    public Matrix abs() {
        double[] abs = new double[entries.length];

        for(int i = 0, size=entries.length; i<size; ++i)
            abs[i] =entries[i].abs();

        return new Matrix(shape, abs);
    }


    /**
     * Converts this dense matrix to an equivalent compressed sparse row (CSR) matrix.
     * @return A CSR matrix equivalent to this matrix.
     */
    public abstract V toCsr();


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public T set(Y value, int... indices) {
        ValidateParameters.ensureValidIndex(shape, indices);
        entries[indices[0]*numCols + indices[1]] = value;
        return (T) this;
    }


    /**
     * Sets column {@code colIdx} to the specified {@code values}.
     * @param values New values of the column to be set.
     * @param colIdx Index of the column to set.
     * @return A reference to this tensor.
     * @throws IndexOutOfBoundsException If {@code values.length != this.numRows}.
     */
    public T setCol(Field<Y>[] values, int colIdx) {
        ValidateParameters.ensureArrayLengthsEq(values.length, this.numRows);

        int rowOffset = 0;
        for(int i=0, size=values.length; i<size; i++) {
            entries[rowOffset + colIdx] = values[i];
            rowOffset += numCols;
        }

        return (T) this;
    }


    /**
     * Sets row {@code rowIdx} to the specified {@code values}.
     * @param values New values of the column to be set.
     * @param rowIdx Index of the row to set.
     * @return A reference to this tensor.
     * @throws IndexOutOfBoundsException If {@code values.length != this.numCols}.
     */
    public T setRow(Field<Y>[] values, int rowIdx) {
        ValidateParameters.ensureValidIndices(numRows, rowIdx);
        ValidateParameters.ensureArrayLengthsEq(values.length, numCols);

        int rowOffset = rowIdx*numCols;
        for(int i=0, size=numCols; i<size; i++) {
            entries[rowOffset + i] = values[i];
        }

        return (T) this;
    }
}
