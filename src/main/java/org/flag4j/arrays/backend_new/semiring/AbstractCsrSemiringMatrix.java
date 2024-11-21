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

package org.flag4j.arrays.backend_new.semiring;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend_new.AbstractTensor;
import org.flag4j.arrays.backend_new.MatrixMixin;
import org.flag4j.arrays.backend_new.SparseMatrixData;
import org.flag4j.linalg.operations.sparse.csr.CsrConversions;
import org.flag4j.linalg.operations.sparse.csr.CsrOps;
import org.flag4j.linalg.operations.sparse.csr.CsrProperties;
import org.flag4j.linalg.operations.sparse.csr.semiring_ops.SemiringCsrMatMult;
import org.flag4j.linalg.operations.sparse.csr.semiring_ops.SemiringCsrOps;
import org.flag4j.linalg.operations.sparse.csr.semiring_ops.SemiringCsrProperties;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.List;

import static org.flag4j.linalg.operations.sparse.SparseUtils.sortCsrMatrix;


/**
 * <p>A real sparse matrix stored in compressed sparse row (CSR) format. The {@link #entries} of this CSR matrix are
 * elements of a {@link Semiring}.
 *
 * <p>The {@link #entries non-zero entries} and non-zero indices of a CSR matrix are mutable but the {@link #shape}
 * and {@link #nnz total number of non-zero entries} is fixed.
 *
 * <p>Sparse matrices allow for the efficient storage of and operations on matrices that contain many zero values.
 *
 * <p>A sparse CSR matrix is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #entries} of the matrix. All other entries in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #rowPointers row pointers} of the non-zero values in the CSR matrix. Has size {@link #numRows numRows + 1}</li>
 *     <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all values in row
 *     {@code i}.
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the CSR matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified. Any operations implemented in this
 * class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.
 *
 * @param <T> Type of this CSR field matrix.
 * @param <U> Type of dense field matrix equivalent to {@code T}.
 * @param <V> Type of vector equivalent to {@code V}.
 * @param <Y> Type of field element of this matrix.
 */
public abstract class AbstractCsrSemiringMatrix<T extends AbstractCsrSemiringMatrix<T, U, V, W>,
        U extends AbstractDenseSemiringMatrix<U, ?, W>,
        V extends AbstractCooSemiringVector<V, ?, ?, U, W>,
        W extends Semiring<W>>
        extends AbstractTensor<T, Semiring<W>[], W>
        implements SemiringTensorMixin<T, U, W>, MatrixMixin<T, U, V, W> {

    /**
     * The zero element for the semiring that this tensor's elements belong to.
     */
    private Semiring<W> zeroElement;
    /**
     * <p>Pointers indicating starting index of each row within the {@link #colIndices} and {@link #entries} arrays.
     * Has length {@link #numRows numRows + 1}.
     *
     * <p>The range [{@code entries[rowPointers[i]], entries[rowPointers[i+1]]}) contains all {@link #entries non-zero entries} within
     * row {@code i}.
     *
     * <p>Similarly, [{@code colIndices[rowPointers[i]], colIndices[rowPointers[i+1]]}) contains all {@link #colIndices column indices}
     * for the entries in row {@code i}.
     * 
     */
    public final int[] rowPointers;
    /**
     * Column indices for non-zero values of this sparse CSR matrix.
     */
    public final int[] colIndices;
    /**
     * Number of non-zero entries in this CSR matrix.
     */
    public final int nnz;
    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;
    /**
     * The sparsity of this matrix.
     */
    private final double sparsity;


    /**
     * Creates a sparse CSR matrix with the specified {@code shape}, non-zero entries, row pointers, and non-zero column indices.
     *
     * @param shape Shape of this tensor.
     * @param entries The non-zero entries of this CSR matrix.
     * @param rowPointers The row pointers for the non-zero values in the sparse CSR matrix.
     * <p>{@code rowPointers[i]} indicates the starting index within {@code entries} and {@code colIndices} of all
     * values in row {@code i}.
     * @param colIndices Column indices for each non-zero value in this sparse CSR matrix. Must satisfy
     * {@code entries.length == colIndices.length}.
     */
    protected AbstractCsrSemiringMatrix(Shape shape, Semiring<W>[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        this.nnz = entries.length;
        this.numRows = shape.get(0);
        this.numCols = shape.get(1);

        sparsity = BigDecimal.valueOf(nnz).divide(new BigDecimal(shape.totalEntries())).doubleValue();

        // Attempt to set the zero element for the semiring.
        this.zeroElement = (entries.length > 0) ? entries[0].getZero() : null;
    }


    /**
     * Constructs a sparse CSR tensor of the same type as this tensor with the specified non-zero entries and indices.
     * @param shape Shape of the matrix.
     * @param entries Non-zero entries of the CSR matrix.
     * @param rowPointers Row pointers for the non-zero values in the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return A sparse CSR tensor of the same type as this tensor with the specified non-zero entries and indices.
     */
    public abstract T makeLikeTensor(Shape shape, W[] entries, int[] rowPointers, int[] colIndices);


    /**
     * Constructs a CSR matrix with the specified shape, non-zero entries, and non-zero indices.
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the CSR matrix.
     * @param rowPointers Row pointers for the non-zero values in the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return A CSR matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    public abstract T makeLikeTensor(Shape shape, List<W> entries, List<Integer> rowPointers, List<Integer> colIndices);


    /**
     * Constructs a dense matrix which is of a similar type to this sparse CSR matrix.
     * @param shape Shape of the dense matrix.
     * @param entries Entries of the dense matrix.
     * @return A dense matrix which is of a similar type to this sparse CSR matrix with the specified {@code shape}
     * and {@code entries}.
     */
    public abstract U makeLikeDenseTensor(Shape shape, W[] entries);


    /**
     * <p>Constructs a sparse COO matrix of a similar type to this sparse CSR matrix.
     * <p>Note: this method constructs a new COO matrix with the specified entries and indices. It does <i>not</i> convert this matrix
     * to a CSR matrix. To convert this matrix to a sparse COO matrix use {@link #toCoo()}.
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero entries of the COO matrix.
     * @param rowIndices Non-zero row indices of the sparse COO matrix.
     * @param colIndices Non-zero column indices of the Sparse COO matrix.
     * @return A sparse COO matrix of a similar type to this sparse CSR matrix.
     */
    public abstract AbstractCooSemiringMatrix<?, U, V, W> makeLikeCooMatrix(
            Shape shape, Semiring<W>[] entries, int[] rowIndices, int[] colIndices);


    /**
     * Gets the sparsity of this matrix as a decimal percentage.
     * That is, the percentage of entries in this matrix that are zero.
     * @return The sparsity of this matrix as a decimal percentage.
     * @see #density()
     */
    public double sparsity() {
        return sparsity;
    }


    /**
     * Gets the density of this matrix as a decimal percentage.
     * That is, the percentage of entries in this matrix that are non-zero.
     * @return The density of this matrix as a decimal percentage.
     * @see #sparsity
     */
    public double density() {
        return 1.0 - sparsity;
    }


    /**
     * Gets the zero element for the semiring of this tensor.
     * @return The zero element for the semiring of this tensor. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(Semiring)} then {@code null} will be returned.
     *
     * @see #setZeroElement(Semiring)
     */
    public W getZeroElement() {
        return (W) zeroElement;
    }


    /**
     * Sets the zero element for the semiring of this tensor.
     * @param zeroElement The zero element of this tensor.
     * @throws IllegalArgumentException If {@code zeroElement} is not an additive identity for the semiring.
     *
     * @see #getZeroElement()
     */
    public void setZeroElement(Semiring<W> zeroElement) {
        if (zeroElement.isZero()) {
            this.zeroElement = zeroElement;
        } else {
            throw new IllegalArgumentException("The provided zeroElement is not an additive identity.");
        }
    }



    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this tensor.
     */
    @Override
    public W get(int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        int row = indices[0];
        int col = indices[1];
        int loc = Arrays.binarySearch(colIndices, rowPointers[row], rowPointers[row+1], col);

        if(loc >= 0) return (W) entries[loc];
        else return (W) zeroElement;
    }


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
    public T set(W value, int... indices) {
        ValidateParameters.validateTensorIndex(shape, indices);
        return set(value, indices[0], indices[1]);
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public T flatten() {
        int[] newRowPointers = new int[2];
        newRowPointers[1] = nnz;
        return makeLikeTensor(
                new Shape(1, shape.totalEntriesIntValueExact()),
                (W[]) entries.clone(),
                newRowPointers,
                colIndices.clone());
    }


    /**
     * Flattens a tensor along the specified axis. Unlike {@link #flatten()}
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public T flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] newRowPointers;
        int[] newColIndices;

        if (axis == 0) {
            // Flatten to a single row.
            newRowPointers = new int[2];
            newRowPointers[1] = nnz;
            newColIndices = new int[nnz];
        } else {
            // Flatten to a single column.
            int flatSize = shape.totalEntriesIntValueExact();
            newColIndices = new int[nnz];  // Set all column indices to 0.
            newRowPointers = new int[flatSize + 1];
        }

        Shape newShape = CsrConversions.flatten(shape, entries, rowPointers, colIndices, axis, newRowPointers, newColIndices);

        return makeLikeTensor(
                new Shape(shape.totalEntriesIntValueExact(), 1),
                (W[]) entries.clone(),
                newRowPointers,
                newColIndices);
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws TensorShapeException If {@code newShape} is not broadcastable to {@link #shape this.shape}.
     */
    @Override
    public T reshape(Shape newShape) {
        return (T) toCoo().reshape(newShape).toCsr();
    }


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor.
     *
     * @return The transpose of this tensor.
     *
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public T T() {
        Semiring<W>[] dest = new Semiring[entries.length];
        int[] destRowPointers = new int[numCols+1];
        int[] destColIndices = new int[entries.length];
        CsrOps.transpose(entries, rowPointers, colIndices, dest, destRowPointers, destColIndices);

        return makeLikeTensor(shape.swapAxes(0, 1), (W[]) dest, destRowPointers, destColIndices);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T add(T b) {
        SparseMatrixData<W> destData = CsrOps.applyBinOpp(
                shape, (W[]) entries, rowPointers, colIndices,
                b.shape, (W[]) b.entries, b.rowPointers, b.colIndices,
                Semiring::add, null);

        return makeLikeTensor(shape, destData.entries(), destData.rowIndices(), destData.colIndices());
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public T elemMult(T b) {
        SparseMatrixData<W> destData = CsrOps.applyBinOpp(
                shape, (W[]) entries, rowPointers, colIndices,
                b.shape, (W[]) b.entries, b.rowPointers, b.colIndices,
                Semiring::mult, null);

        return makeLikeTensor(shape, destData.entries(), destData.rowIndices(), destData.colIndices());
    }


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
    public AbstractDenseSemiringTensor<?, W> tensorDot(T src2, int[] aAxes, int[] bAxes) {
        // TODO: Implement this method. Need to wait for a concrete implementation of AbstractDenseSemiringTensor
        return null;
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.
     *
     * @param axis1 First axis for 2D sub-array.
     * @param axis2 Second axis for 2D sub-array.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException  If {@code axis1 == axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e. the axes are equal or the tensor does not have the same length along the two axes.)
     */
    @Override
    public T tensorTr(int axis1, int axis2) {
        ValidateParameters.ensureNotEquals(axis1, axis2);
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);

        return (T) makeLikeTensor(new Shape(1, 1), (W[]) new Semiring[]{tr()}, new int[]{0}, new int[]{0});
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
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        if(axis1 == axis2) return copy();
        return T();
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
        if(axes.length != 2) {
            throw new IllegalArgumentException("Cannot transpose axes "
                    + Arrays.toString(axes) + " for a tensor of rank " + rank);
        }

        return T(axes[0], axes[1]);
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
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     *
     * <p>Same as {@link #trace()}.
     *
     * @return The trace of this matrix.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public W tr() {
        W tr = (W) SemiringCsrOps.trace(entries, rowPointers, colIndices);

        return (tr == null) ? (W) zeroElement : tr;
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
        return SemiringCsrProperties.isTriU(shape, entries, rowPointers, colIndices);
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
        return SemiringCsrProperties.isTriL(shape, entries, rowPointers, colIndices);
    }


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     */
    @Override
    public boolean isI() {
        return SemiringCsrProperties.isIdentity(shape, entries, rowPointers, colIndices);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number
     *                                of rows in matrix {@code b}.
     * @see #multToSparse(AbstractCsrSemiringMatrix)
     */
    @Override
    public U mult(T b) {
        Shape destShape = new Shape(numRows, b.numCols);
        W[] destArray = (W[]) new Semiring[numRows*b.numCols];

        SemiringCsrMatMult.standard(
                shape, entries, rowPointers, colIndices, b.shape,
                b.entries, b.rowPointers, b.colIndices,
                destArray, zeroElement);

        return makeLikeDenseTensor(shape, destArray);
    }


    /**
     * <p>Computes the matrix multiplication between two sparse CSR matrices and stores the result in a sparse matrix.
     * <p>Warning: this method should be used with caution as sparse-sparse matrix multiplication may result in a dense matrix.
     * In such a case, this method will likely be significantly slower than {@link #mult(AbstractCsrSemiringMatrix)}.
     * @param b
     * @return
     */
    public T multToSparse(T b) {
        SparseMatrixData<W> data = SemiringCsrMatMult.standardToSparse(
                shape, entries, rowPointers, colIndices, b.shape,
                b.entries, b.rowPointers, b.colIndices);

        return makeLikeTensor(data.shape(), data.entries(), data.rowIndices(), data.colIndices());
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
    public U multTranspose(T b) {
        ValidateParameters.ensureEquals(numCols, b.numCols);
        return mult(b.T());
    }


    /**
     * Stacks matrices along columns. <br>
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(MatrixMixin, int)
     * @see #augment(T)
     */
    @Override
    public T stack(T b) {
        return (T) toCoo().stack(b.toCoo()).toCsr();
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(T)
     * @see #stack(MatrixMixin, int)
     */
    @Override
    public T augment(T b) {
        return (T) toCoo().augment(b.toCoo()).toCsr();
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    @Override
    public T augment(V b) {
        return (T) toCoo().augment(b).toCsr();
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
        CsrOps.swapRows(entries, rowPointers, colIndices, rowIndex1, rowIndex2);
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
        CsrOps.swapCols(entries, rowPointers, colIndices, colIndex1, colIndex2);
        return (T) this;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isSymmetric() {
        return CsrProperties.isSymmetric(shape, entries, rowPointers, colIndices);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        // For a semiring matrix, same as isSymmetric.
        return isSymmetric();
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        if(isSquare()) return mult(T()).isI();
        else return false;
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
        return (T) toCoo().removeRow(rowIndex).toCsr();
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeRows(int... rowIndices) {
        return (T) toCoo().removeRows(rowIndices).toCsr();
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCol(int colIndex) {
        return (T) toCoo().removeCol(colIndex).toCsr();
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public T removeCols(int... colIndices) {
        return (T) toCoo().removeCols(colIndices).toCsr();
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
        return (T) toCoo().setSliceCopy(values.toCoo(), rowStart, colStart).toCsr();
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
        SparseMatrixData<Semiring<W>> sliceData = CsrOps.getSlice(
                entries, rowPointers, colIndices,
                rowStart, rowEnd, colStart, colEnd);
        return makeLikeTensor(sliceData.shape(), (List<W>) sliceData.entries(),
                sliceData.rowIndices(), sliceData.colIndices());
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
    public T set(W value, int row, int col) {
        // Ensure indices are in bounds.
        ValidateParameters.validateTensorIndex(shape, row, col);
        Semiring<W>[] newEntries;
        int[] newRowPointers = rowPointers.clone();
        int[] newColIndices;
        boolean found = false; // Flag indicating an element already exists in this matrix at the specified row and col.
        int loc = -1;

        if(rowPointers[row] < rowPointers[row+1]) {
            int start = rowPointers[row];
            int stop = rowPointers[row+1];

            loc = Arrays.binarySearch(colIndices, start, stop, col);
            found = loc >= 0;
        }

        if(found) {
            newEntries = entries.clone();
            newEntries[loc] = value;
            newRowPointers = rowPointers.clone();
            newColIndices = colIndices.clone();
        } else {
            loc = -loc - 1; // Compute insertion index as specified by Arrays.binarySearch.
            newEntries = new Field[entries.length + 1];
            newColIndices = new int[entries.length + 1];

            CsrOps.insertNewValue(
                    entries, rowPointers, colIndices,
                    newEntries, newRowPointers, newColIndices,
                    row, col, loc, value);
        }

        return makeLikeTensor(shape, (W[]) newEntries, newRowPointers, newColIndices);
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
        return (T) toCoo().getTriU(diagOffset).toCsr();
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
        return (T) toCoo().getTriL(diagOffset).toCsr();
    }


    /**
     * Creates a deep copy of this tensor.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeTensor(shape, entries.clone());
    }


    /**
     * Computes the Hermitian transpose of this matrix.
     *
     * @return The Hermitian transpose of this matrix.
     */
    @Override
    public T H() {
        return T();
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     */
    public void sortIndices() {
        sortCsrMatrix(entries, rowPointers, colIndices);
    }


    /**
     * <p>Converts this sparse CSR matrix to an equivalent dense matrix.
     * 
     * <p>The zero entries of this CSR matrix will be attempted to be filled with a zero value if it could be determined during 
     * construction of this sparse CSR matrix. If the zero value could not be determined the zero entries will be filled with 
     * {@code null} (this only happens when {@code nnz==0}). To avoid this, the zero element of the semiring for this 
     * matrix can be set explicitly using {@link #setZeroElement(Semiring)}.
     * 
     * @return A dense matrix which is equivalent to this sparse CSR matrix.
     */
    public U toDense() {
        Semiring<W>[] dest = new Semiring[shape.totalEntriesIntValueExact()];
        CsrConversions.toDense(shape, entries, rowPointers, colIndices, dest, zeroElement);
        return makeLikeDenseTensor(shape, (W[]) dest);
    }


    /**
     * Converts this sparse CSR matrix to an equivalent sparse COO matrix.
     * @return A sparse COO matrix equivalent to this sparse CSR matrix.
     */
    public AbstractCooSemiringMatrix toCoo() {
        Semiring<W>[] cooEntries = new Semiring[nnz];
        int[] cooRowIndices = new int[nnz];
        int[] cooColIndices = new int[nnz];
        CsrConversions.toCoo(shape, entries, rowPointers, colIndices, cooEntries, cooRowIndices, cooColIndices);
        return makeLikeCooMatrix(shape, cooEntries, cooRowIndices, cooColIndices);
    }


    /**
     * Converts this CSR matrix to an equivalent sparse COO tensor.
     * @return An sparse COO tensor equivalent to this CSR matrix.
     */
    public AbstractTensor<?, Semiring<W>[], W> toTensor() {
        return toCoo().toTensor();
    }


    /**
     * Converts this CSR matrix to an equivalent COO tensor with the specified shape.
     * @param newShape New shape for the COO tensor. Can be any rank but must be broadcastable to {@link #shape this.shape}.
     * @return A COO tensor equivalent to this CSR matrix which has been reshaped to {@code newShape}
     */
    public AbstractTensor<?, Semiring<W>[], W> toTensor(Shape shape) {
        return toCoo().toTensor();
    }


    /**
     * Converts this sparse CSR matrix to an equivalent vector. If this matrix is not a row or column vector it will be flattened
     * before conversion.
     * @return A vector equivalent to this CSR matrix.
     */
    public V toVector() {
        return (V) toCoo().toVector();
    }
}
