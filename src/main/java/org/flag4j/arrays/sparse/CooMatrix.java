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

package org.flag4j.arrays.sparse;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.primitive.AbstractDoubleTensor;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.common.real.RealProperties;
import org.flag4j.linalg.operations.dense.real.RealDenseTranspose;
import org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseOperations;
import org.flag4j.linalg.operations.dense_sparse.coo.real.RealDenseSparseMatrixOperations;
import org.flag4j.linalg.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixOperations;
import org.flag4j.linalg.operations.sparse.coo.CooDataSorter;
import org.flag4j.linalg.operations.sparse.coo.real.*;
import org.flag4j.linalg.operations.sparse.coo.real_complex.RealComplexSparseMatrixMultiplication;
import org.flag4j.linalg.operations.sparse.coo.real_complex.RealComplexSparseMatrixOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A real sparse matrix stored in coordinate list (COO) format. The {@link #entries} of this COO tensor are
 * primitive doubles.
 *
 * <p>The {@link #entries non-zero entries} and non-zero indices of a COO matrix are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.
 *
 * <p>COO matrices are well-suited for incremental matrix construction and modification but may not have ideal efficiency for matrix
 * operations like matrix multiplication. For heavy computations, it may be better to construct a matrix as a {@code CooMatrix} then
 * convert to a {@link CsrMatrix} (using {@link #toCsr()}) as CSR (compressed sparse row) matrices are generally better suited for
 * efficient
 * matrix operations.
 *
 * <p>A sparse COO matrix is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #entries} of the matrix. All other entries in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #rowIndices row indices} of the non-zero values in the sparse matrix.</li>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Some operations on sparse tensors behave differently than on dense tensors. For instance, {@link #add(double)} will not
 * add the scalar to all entries of the tensor since this would cause catastrophic loss of sparsity. Instead, such non-zero preserving
 * element-wise operations only act on the non-zero entries of the sparse tensor as to not affect the sparsity.
 *
 * <p>Note: many operations assume that the entries of the COO matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified. Any operations implemented in this
 * class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.
 */
public class CooMatrix extends AbstractDoubleTensor<CooMatrix>
        implements MatrixMixin<CooMatrix, Matrix, CooVector, Double> {

    /**
     * Row indices for non-zero value of this sparse COO matrix.
     */
    public final int[] rowIndices;
    /**
     * column indices for non-zero value of this sparse COO matrix.
     */
    public final int[] colIndices;
    /**
     * Number of non-zero entries in this COO matrix.
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
    private double sparsity = -1;


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooMatrix(Shape shape, double[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);
        ValidateParameters.ensureArrayLengthsEq(entries.length, rowIndices.length, colIndices.length);
        ValidateParameters.ensureTrue(
                shape.totalEntries().compareTo(BigInteger.valueOf(entries.length)) >= 0,
                "Shape " + shape + " cannot hold " + entries.length + "entries.");

        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        nnz = entries.length;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooMatrix(int numRows, int numCols, double[] entries, int[] rowIndices, int[] colIndices) {
        super(new Shape(numRows, numCols), entries);
        ValidateParameters.ensureRank(shape, 2);
        ValidateParameters.ensureArrayLengthsEq(entries.length, rowIndices.length, colIndices.length);
        ValidateParameters.ensureTrue(
                shape.totalEntries().compareTo(BigInteger.valueOf(entries.length)) >= 0,
                "Shape " + shape + " cannot hold " + entries.length + "entries.");

        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        nnz = entries.length;
        this.numRows = numRows;
        this.numCols = numCols;
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooMatrix(Shape shape, List<Double> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(shape, ArrayUtils.fromDoubleList(entries));
        ValidateParameters.ensureArrayLengthsEq(entries.size(), rowIndices.size(), colIndices.size());
        ValidateParameters.ensureTrue(
                shape.totalEntries().compareTo(BigInteger.valueOf(entries.size())) >= 0,
                "Shape " + shape + " cannot hold " + entries.size() + "entries.");
        ValidateParameters.ensureRank(shape, 2);

        this.rowIndices = ArrayUtils.fromIntegerList(rowIndices);
        this.colIndices = ArrayUtils.fromIntegerList(colIndices);
        ValidateParameters.ensureArrayLengthsEq(super.entries.length, this.rowIndices.length, this.colIndices.length);

        nnz = super.entries.length;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Constructs a square zero matrix with the specified size.
     * @param size Size of the square zero matrix to construct.
     */
    public CooMatrix(int size) {
        super(new Shape(size, size), new double[0]);
        numCols = numRows = size;
        nnz = 0;
        rowIndices = new int[0];
        colIndices = new int[0];
    }


    /**
     * Constructs a zero matrix with the specified shape.
     * @param rows The number of rows in the zero matrix.
     * @param cols The number of columns in the zero matrix.
     */
    public CooMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new double[0]);
        numRows = rows;
        numCols = cols;
        nnz = 0;
        rowIndices = new int[0];
        colIndices = new int[0];
    }


    /**
     * Constructs a zero matrix with the specified shape.
     * @param shape
     */
    public CooMatrix(Shape shape) {
        super(shape, new double[0]);
        ValidateParameters.ensureRank(shape, 2);
        numRows = shape.get(0);
        numCols = shape.get(1);
        nnz = 0;
        rowIndices = new int[0];
        colIndices = new int[0];
    }


    /**
     * Creates a square sparse COO matrix with the specified size, non-zero entries, and non-zero indices.
     * @param size Size of the square matrix.
     * @param entries Non-zero entries of the sparse matrix.
     * @param rowIndices Row indices of the non-zero entries in the matrix.
     * @param colIndices Column indices of the non-zero entries in the matrix.
     */
    public CooMatrix(int size, double[] entries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size), entries);
        ValidateParameters.ensureArrayLengthsEq(entries.length, rowIndices.length, colIndices.length);
        ValidateParameters.ensureTrue(
                shape.totalEntries().compareTo(BigInteger.valueOf(entries.length)) >= 0,
                "Shape " + shape + " cannot hold " + entries.length + "entries.");

        this.nnz = entries.length;
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        numRows = numCols = size;
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooMatrix(Shape shape, int[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, new double[entries.length]);
        ValidateParameters.ensureArrayLengthsEq(entries.length, rowIndices.length, colIndices.length);
        ValidateParameters.ensureTrue(
                shape.totalEntries().compareTo(BigInteger.valueOf(entries.length)) >= 0,
                "Shape " + shape + " cannot hold " + entries.length + "entries.");
        ValidateParameters.ensureRank(shape, 2);

        ArrayUtils.asDouble(entries, super.entries);
        this.rowIndices = rowIndices;
        this.colIndices = colIndices;
        nnz = entries.length;
        numRows = shape.get(0);
        numCols = shape.get(1);
    }


    /**
     * Constructs a copy of a real sparse COO matrix.
     * @param b
     */
    public CooMatrix(CooMatrix b) {
        super(b.shape, b.entries.clone());
        rowIndices = b.rowIndices.clone();
        colIndices = b.colIndices.clone();
        nnz = b.nnz;
        numRows = b.numRows;
        numCols = b.numCols;
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
    public Matrix tensorDot(CooMatrix src2, int[] aAxes, int[] bAxes) {
        CooTensor t1 = new CooTensor(
                shape, entries, RealDenseTranspose.blockedIntMatrix(
                        new int[][]{rowIndices, colIndices}));
        CooTensor t2 = new CooTensor(
                src2.shape, src2.entries, RealDenseTranspose.blockedIntMatrix(
                new int[][]{src2.rowIndices, src2.colIndices}));

        return RealCooTensorDot.tensorDot(t1, t2, aAxes, bAxes).toMatrix();
    }


    /**
     * Constructs a sparse COO matrix of the same type as this tensor with the given the shape and entries and indices copied from
     * this matrix.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A matrix of the same type as this matrix with the given the shape and entries.
     */
    @Override
    public CooMatrix makeLikeTensor(Shape shape, double[] entries) {
        return new CooMatrix(shape, entries, rowIndices.clone(), colIndices.clone());
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
    public CooMatrix T(int axis1, int axis2) {
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
    public CooMatrix T(int... axes) {
        ValidateParameters.ensureArrayLengthsEq(2, axes.length);
        ValidateParameters.ensurePermutation(axes);
        return T();
    }


    /**
     * The sparsity of this sparse tensor. That is, the percentage of elements in this tensor which are zero as a decimal.
     *
     * @return The density of this sparse tensor.
     */
    public double sparsity() {
        // Check if the sparsity has already been computed.
        if (this.sparsity < 0) {
            BigInteger totalEntries = totalEntries();
            BigDecimal sparsity = new BigDecimal(totalEntries).subtract(BigDecimal.valueOf(nnz));
            sparsity = sparsity.divide(new BigDecimal(totalEntries), RoundingMode.HALF_UP);

            this.sparsity = sparsity.doubleValue();
        }

        return sparsity;
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor equivalent to this sparse tensor.
     * @throws ArithmeticException If the total number of entries in this sparse matrix does not fit into an int.
     */
    public Matrix toDense() {
        double[] entries = new double[totalEntries().intValueExact()];
        int row;
        int col;

        for(int i = 0; i< nnz; i++) {
            row = rowIndices[i];
            col = colIndices[i];

            entries[row*numCols + col] = this.entries[i];
        }

        return new Matrix(shape, entries);
    }


    /**
     * Converts this COO matrix to an equivalent CSR matrix.
     * @return A CSR matrix equivalent to this matrix.
     */
    public CsrMatrix toCsr() {
        int[] csrRowPointers = new int[numRows + 1];

        // Copy the non-zero entries anc column indices. Count number of entries per row.
        for(int i=0, size=entries.length; i<size; i++)
            csrRowPointers[rowIndices[i] + 1]++;

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0, size=numRows; i<size; i++)
            csrRowPointers[i+1] += csrRowPointers[i];

        return new CsrMatrix(shape, entries.clone(), csrRowPointers, colIndices.clone());
    }


    /**
     * Sorts the indices of this tensor in lexicographical order while maintaining the associated value for each index.
     * @return A reference to this matrix.
     */
    public CooMatrix sortIndices() {
        CooDataSorter.wrap(entries, rowIndices, colIndices).sparseSort().unwrap(entries, rowIndices, colIndices);
        return this;
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this matrix.
     */
    @Override
    public Double get(int... indices) {
        ValidateParameters.ensureEquals(indices.length, 2);
        ValidateParameters.ensureIndexInBounds(numRows, indices[0]);
        ValidateParameters.ensureIndexInBounds(numCols, indices[1]);

        return get(indices[0], indices[1]);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param indices Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public CooMatrix set(Double value, int... indices) {
        ValidateParameters.ensureArrayLengthsEq(2, indices.length);
        ValidateParameters.ensureIndexInBounds(numRows, indices[0]);
        ValidateParameters.ensureIndexInBounds(numCols, indices[1]);

        return RealSparseMatrixGetSet.matrixSet(this, indices[0], indices[1], value);
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public CooMatrix flatten() {
        int[] destIndices = new int[entries.length];

        for(int i = 0; i < entries.length; i++)
            destIndices[i] = shape.getFlatIndex(rowIndices[i], colIndices[i]);

        return new CooMatrix(shape, entries.clone(), new int[entries.length], destIndices);
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public CooMatrix flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] dims = {1, 1};
        dims[1-axis] = this.totalEntries().intValueExact();

        int[] rowIndices = axis==1 ? this.rowIndices.clone() : new int[this.rowIndices.length];
        int[] colIndices = axis==0 ? this.colIndices.clone() : new int[this.colIndices.length];

        return new CooMatrix(new Shape(dims), entries.clone(), rowIndices, colIndices);
    }


    /**
     * Copies and reshapes this tensor.
     *
     * @param newShape New shape for the tensor.
     *
     * @return A copy of this tensor with the new shape.
     *
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code newShape} is not broadcastable to {@link #shape this.shape}.
     */
    @Override
    public CooMatrix reshape(Shape newShape) {
        ValidateParameters.ensureBroadcastable(shape, newShape);
        int oldColCount = shape.get(1);
        int newColCount = newShape.get(1);

        // Initialize new COO structures with the same size as the original.
        int[] newRowIndices = new int[rowIndices.length];
        int[] newColIndices = new int[colIndices.length];

        for (int i = 0; i < rowIndices.length; i++) {
            int flatIndex = rowIndices[i]*oldColCount + colIndices[i];
            newRowIndices[i] = flatIndex / newColCount;
            newColIndices[i] = flatIndex % newColCount;
        }

        return new CooMatrix(newShape, entries.clone(), newRowIndices, newColIndices);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CooMatrix add(CooMatrix b) {
        return RealSparseMatrixOperations.add(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    @Override
    public CooMatrix sub(CooMatrix b) {
        return RealSparseMatrixOperations.sub(this, b);
    }


    /**
     * Computes the conjugate transpose of a tensor by exchanging the first and last axes of this tensor and conjugating the
     * exchanged values.
     *
     * @return The conjugate transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public CooMatrix H() {
        return T();
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmin() {
        int idx = RealProperties.argmin(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        int idx = RealProperties.argmax(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        int idx = RealProperties.argminAbs(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        int idx = RealProperties.argmaxAbs(entries);
        return new int[]{rowIndices[idx], colIndices[idx]};
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
    public CooMatrix elemMult(CooMatrix b) {
        return RealSparseMatrixOperations.elemMult(this, b);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.
     *
     * <p>Note: for a matrix, the {@link #tr()} method is preferred.
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.
     *
     * @param axis1 First axis for 2D sub-array.
     * @param axis2 Second axis for 2D sub-array.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}. This will be a tensor of rank
     * {@code this.getRank() - 2} with the same shape as this tensor but with {@code axis1} and {@code axis2} removed.
     *
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException  If {@code axis1 == @code axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     *                                   (i.e. the axes are equal or the tensor does not have the same length along the two axes.)
     */
    @Override
    public CooMatrix tensorTr(int axis1, int axis2) {
        ValidateParameters.ensureNotEquals(axis1, axis2);
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);

        return new CooMatrix(new Shape(1, 1), new double[]{tr()}, new int[]{0}, new int[]{0});
    }


    /**
     * Computes the product of all non-zero values in this tensor.
     *
     * @return The product of all non-zero values in this tensor.
     */
    @Override
    public Double prod() {
        return super.prod(); // Overrides method from super class to emphasize it operates only on the non-zero values.
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
    public CooMatrix T() {
        CooMatrix transpose = new CooMatrix(
                shape.swapAxes(0, 1),
                entries.clone(),
                colIndices.clone(),
                rowIndices.clone()
        );

        transpose.sortIndices(); // Ensure the indices are sorted correctly.

        return transpose;
    }


    /**
     * Computes the element-wise reciprocals of non-zero values of this tensor.
     *
     * @return A tensor containing the reciprocals of the non-zero values of this tensor.
     */
    @Override
    public CooMatrix recip() {
        return super.recip(); // Overrides method from super class to emphasize it operates on the non-zero values in the tensor.
    }


    /**
     * Adds a scalar value to each non-zero element of this tensor.
     *
     * @param b Value to add to each non-zero entry of this tensor.
     *
     * @return The result of adding the specified scalar value to each non-zero entry of this tensor.
     */
    @Override
    public CooMatrix add(double b) {
        return super.add(b); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * Adds a scalar value to each non-zero element of this tensor.
     *
     * @param b Value to add to each non-zero entry of this tensor.
     *
     * @return The result of adding the specified scalar value to each non-zero entry of this tensor.
     */
    public CooCMatrix add(Complex128 b) {
        // Overrides method from super class to emphasize it only operates on the non-zero values.
        return new CooCMatrix(shape, RealFieldDenseOperations.add(entries, b), rowIndices.clone(), colIndices.clone());
    }


    /**
     * Subtracts a scalar value from each non-zero element of this tensor.
     *
     * @param b Value to subtract from each non-zero entry of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each non-zero entry of this tensor.
     */
    @Override
    public CooMatrix sub(double b) {
        return super.sub(b); // Overrides method from super class to emphasize it operates only on the non-zero values.
    }


    /**
     * <p>Computes the element-wise quotient between two tensors.
     * <p><b>Warning</b>: This method is not supported for sparse matrices. If called on a sparse matrix,
     * an {@link UnsupportedOperationException} will be thrown. Element-wise division is undefined for sparse matrices as it
     * would almost certainly result in a division by zero.
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    @Override
    public CooMatrix div(CooMatrix b) {
        throw new UnsupportedOperationException("Cannot compute element-wise division of two sparse matrices.");
    }


    /**
     * Subtracts a scalar value from each non-zero element of this tensor.
     *
     * @param b Value to subtract from each non-zero entry of this tensor.
     *
     * @return The result of subtracting the specified scalar value from each non-zero entry of this tensor.
     */
    public CooCMatrix sub(Complex128 b) {
        // Overrides method from super class to emphasize it operates only on the non-zero values.
        return new CooCMatrix(shape, RealFieldDenseOperations.sub(entries, b), rowIndices.clone(), colIndices.clone());
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
     * Gets the element of this matrix at this specified {@code row} and {@code col}.
     *
     * @param row Row index of the item to get from this matrix.
     * @param col Column index of the item to get from this matrix.
     *
     * @return The element of this matrix at the specified index.
     */
    @Override
    public Double get(int row, int col) {
        ValidateParameters.validateTensorIndex(shape, row, col);
        return RealSparseMatrixGetSet.matrixGet(this, row, col);
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
    public Double tr() {
        double trace = 0;

        for(int i=0, size=entries.length; i<size; i++)
            if(rowIndices[i]==colIndices[i]) trace += entries[i]; // Then entry on the diagonal.

        return trace;
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
        for(int i=0, size=entries.length; i<size; i++)
            if(rowIndices[i] > colIndices[i] && entries[i] != 0) return false; // Then entry is not in upper triangle.

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
        for(int i=0, size=entries.length; i<size; i++)
            if(rowIndices[i] < colIndices[i]&& entries[i] != 0) return false; // Then entry is not in lower triangle.

        return true;
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
        return RealSparseMatrixProperties.isIdentity(this);
    }


    /**
     * Checks that this matrix is close to the identity matrix.
     *
     * @return True if this matrix is approximately the identity matrix.
     *
     * @see #isI()
     */
    public boolean isCloseToI() {
        return RealSparseMatrixProperties.isCloseToIdentity(this);
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     *
     * @throws LinearAlgebraException If this matrix is not square.
     */
    public Double det() {
        return toDense().det();
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
    public Matrix mult(CooMatrix b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);

        return new Matrix(numRows, b.numCols,
                RealSparseMatrixMultiplication.standard(
                    entries, rowIndices, colIndices, shape,
                    b.entries, b.rowIndices, b.colIndices, b.shape
                )
        );
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
    public CMatrix mult(CooCMatrix b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);

        return new CMatrix(numRows, b.numCols,
                RealComplexSparseMatrixMultiplication.standard(
                        entries, rowIndices, colIndices, shape,
                        b.entries, b.rowIndices, b.colIndices, b.shape
                )
        );
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
    public Matrix multTranspose(CooMatrix b) {
        ValidateParameters.ensureEquals(numCols, b.numCols);
        return mult(b.T());
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param b Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix b.
     *
     * @throws IllegalArgumentException If this matrix and b have different shapes.
     */
    public Double fib(CooMatrix b) {
        return T().mult(b).tr();
    }


    /**
     * Stacks matrices along columns. <br>
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(MatrixMixinOld, int)
     * @see #augment(CooMatrix)
     */
    public CooMatrix stack(CooMatrix b) {
        ValidateParameters.ensureEquals(numCols, b.numCols);

        Shape destShape = new Shape(numRows+b.numRows, numCols);
        double[] destEntries = new double[entries.length + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);

        // Copy row indices.
        int[] shiftedRowIndices = ArrayUtils.shift(numRows, b.rowIndices.clone());
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(shiftedRowIndices, 0, destRowIndices, rowIndices.length, b.rowIndices.length);

        // Copy column indices.
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(b.colIndices, 0, destColIndices, colIndices.length, b.colIndices.length);

        return new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(CooMatrix) 
     * @see #stack(MatrixMixinOld, int)
     */
    @Override
    public CooMatrix augment(CooMatrix b) {
        ValidateParameters.ensureEquals(numRows, b.numRows);

        Shape destShape = new Shape(numRows, numCols + b.numCols);
        double[] destEntries = new double[entries.length + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);

        // Copy row indices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(b.rowIndices, 0, destRowIndices, rowIndices.length, b.rowIndices.length);

        // Copy column indices (with shifts if appropriate).
        int[] shifted = b.colIndices.clone();
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(ArrayUtils.shift(numCols, shifted), 0,
                destColIndices, colIndices.length, b.colIndices.length);

        CooMatrix dest = new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure indices are sorted properly.

        return dest;
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    @Override
    public CooMatrix augment(CooVector b) {
        ValidateParameters.ensureEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        double[] destEntries = new double[entries.length + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy values and indices from this matrix.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy values and indices from vector.
        System.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        Arrays.fill(destRowIndices, entries.length, destRowIndices.length, numRows);
        System.arraycopy(b.indices, 0, destColIndices, entries.length, b.entries.length);

        return new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
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
    public CooMatrix swapRows(int rowIndex1, int rowIndex2) {
        return RealSparseMatrixManipulations.swapRows(this, rowIndex1, rowIndex2);
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
    public CooMatrix swapCols(int colIndex1, int colIndex2) {
        return RealSparseMatrixManipulations.swapCols(this, colIndex1, colIndex2);
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
        return RealSparseMatrixProperties.isSymmetric(this);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    @Override
    public boolean isHermitian() {
        return isSymmetric();
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     *
     * @see #isSymmetric()
     */
    public boolean isAntiSymmetric() {
        return RealSparseMatrixProperties.isAntiSymmetric(this);
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        if(isSquare()) return this.mult(this.T()).round().isI();
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
    public CooMatrix removeRow(int rowIndex) {
        return RealSparseMatrixManipulations.removeRow(this, rowIndex);
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CooMatrix removeRows(int... rowIndices) {
        return RealSparseMatrixManipulations.removeRows(this, rowIndices);
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CooMatrix removeCol(int colIndex) {
        return RealSparseMatrixManipulations.removeCol(this, colIndex);
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public CooMatrix removeCols(int... colIndices) {
        return RealSparseMatrixManipulations.removeCols(this, colIndices);
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
    public CooMatrix setSliceCopy(CooMatrix values, int rowStart, int colStart) {
        return RealSparseMatrixGetSet.setSlice(this, values, rowStart, colStart);
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
    public CooMatrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        return RealSparseMatrixGetSet.getSlice(this, rowStart, rowEnd, colStart, colEnd);
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
    public CooMatrix set(Double value, int row, int col) {
        ValidateParameters.ensureValidArrayIndices(numRows, row);
        ValidateParameters.ensureValidArrayIndices(numCols, col);
        return RealSparseMatrixGetSet.matrixSet(this, row, col, value);
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
    public CooMatrix getTriU(int diagOffset) {
        int sizeEst = nnz / 2; // Estimate the number of non-zero entries.
        List<Double> triuEntries = new ArrayList<>(sizeEst);
        List<Integer> triuRowIndices = new ArrayList<>(sizeEst);
        List<Integer> triuColIndices = new ArrayList<>(sizeEst);

        for(int i=0, size=nnz; i<size; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];

            if(col >= row) {
                triuEntries.add(entries[i]);
                triuRowIndices.add(row);
                triuColIndices.add(col);
            }
        }

        return new CooMatrix(shape, triuEntries, triuRowIndices, triuColIndices);
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
    public CooMatrix getTriL(int diagOffset) {
        int sizeEst = nnz / 2; // Estimate the number of non-zero entries.
        List<Double> trilEntries = new ArrayList<>(sizeEst);
        List<Integer> trilRowIndices = new ArrayList<>(sizeEst);
        List<Integer> trilColIndices = new ArrayList<>(sizeEst);

        for(int i=0, size=nnz; i<size; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];

            if(col <= row) {
                trilEntries.add(entries[i]);
                trilRowIndices.add(row);
                trilColIndices.add(col);
            }
        }

        return new CooMatrix(shape, trilEntries, trilRowIndices, trilColIndices);
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
    public Vector mult(CooVector b) {
        double[] dest = RealSparseMatrixMultiplication.standardVector(
                entries, rowIndices, colIndices, shape,
                b.entries, b.indices
        );
        return new Vector(dest);
    }


    /**
     * Converts this matrix to an equivalent vector. If this matrix is not shaped as a row/column vector,
     * it will first be flattened then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public CooVector toVector() {
        int[] destIndices = new int[nnz];

        for(int i=0, size=nnz; i<size; i++)
            destIndices[i] = rowIndices[i]*colIndices[i];

        return new CooVector(shape.totalEntriesIntValueExact(), entries.clone(), destIndices);
    }


    /**
     * Converts this sparse COO matrix to an equivalent {@link CooTensor sparse COO tensor}.
     * @return A {@link CooTensor sparse COO tensor} equivalent to this sparse COO matrix.
     */
    public CooTensor toTensor() {
        int[][] tensorIndices = {rowIndices.clone(),  colIndices.clone()};
        return new CooTensor(shape, entries, RealDenseTranspose.blockedIntMatrix(tensorIndices));
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
    public CooVector getRow(int rowIdx) {
        return RealSparseMatrixGetSet.getRow(this, rowIdx);
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
    public CooVector getRow(int rowIdx, int colStart, int colEnd) {
        return RealSparseMatrixGetSet.getRow(this, rowIdx, colStart, colEnd);
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
    public CooVector getCol(int colIdx) {
        return RealSparseMatrixGetSet.getCol(this, colIdx);
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
    public CooVector getCol(int colIdx, int rowStart, int rowEnd) {
        return RealSparseMatrixGetSet.getCol(this, colIdx, rowStart, rowEnd);
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal entries of this matrix.
     */
    public CooVector getDiag() {
        List<Double> destEntries = new ArrayList<>();
        List<Integer> destIndices = new ArrayList<>();

        for(int i=0, size=nnz; i<size; i++) {
            if(rowIndices[i]==colIndices[i]) {
                // Then entry on the diagonal.
                destEntries.add(entries[i]);
                destIndices.add(rowIndices[i]);
            }
        }

        return new CooVector(
                Math.min(numRows, numCols),
                ArrayUtils.fromDoubleList(destEntries),
                ArrayUtils.fromIntegerList(destIndices)
        );
    }


    /**
     * Gets the elements of this matrix along the specified diagonal.
     *
     * @param diagOffset The diagonal to get within this matrix.
     * <ul>
     *     <li>If {@code diagOffset == 0}: Then the elements of the principle diagonal are collected.</li>
     *     <li>If {@code diagOffset < 0}: Then the elements of the sub-diagonal {@code diagOffset} below the principle diagonal
     *     are collected.</li>
     *     <li>If {@code diagOffset > 0}: Then the elements of the super-diagonal {@code diagOffset} above the principle diagonal
     *     are collected.</li>
     * </ul>
     *
     * @return The elements of the specified diagonal as a vector.
     */
    @Override
    public CooVector getDiag(int diagOffset) {
        // Validate diagOffset is within the valid range
        ValidateParameters.ensureInRange(diagOffset, -(numRows-1),
                numCols-1, "diagOffset");

        // Calculate the length of the diagonal.
        int length;
        if (diagOffset >= 0)
            length = Math.min(numRows, numCols - diagOffset);
        else
            length = Math.min(numRows + diagOffset, numCols);

        // Determine the starting row index based on diagOffset
        int startRow = diagOffset >= 0 ? 0 : -diagOffset;

        // Lists to store positions and values of non-zero diagonal elements
        List<Integer> idxList = new ArrayList<>();
        List<Double> entriesList = new ArrayList<>();

        // Iterate over non-zero entries in the COO matrix
        for (int i = 0; i < nnz; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];

            // Check if the current element is on the specified diagonal
            if (col - row == diagOffset) {
                int pos = row - startRow; // Position in the diagonal vector
                idxList.add(pos);
                entriesList.add(entries[i]);
            }
        }

        // Convert lists to arrays.
        int nnzDiag = idxList.size();
        int[] diagIndices = new int[nnzDiag];
        double[] diagEntries = new double[nnzDiag];
        for (int i = 0; i < nnzDiag; i++) {
            diagIndices[i] = idxList.get(i);
            diagEntries[i] = entriesList.get(i);
        }

        // Create and return the sparse vector representing the diagonal
        return new CooVector(length, diagEntries, diagIndices);
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A copy of this matrix with the specified column set to {@code values}.
     *
     * @throws IllegalArgumentException If the values vector has a different length than the number of rows of this matrix.
     * @throws IndexOutOfBoundsException If {@code colIndex < 0 || colIndex >= this.numCols}.
     */
    public CooMatrix setCol(CooVector values, int colIndex) {
        return RealSparseMatrixGetSet.setCol(this, colIndex, values);
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A copy of this matrix with the specified row set to {@code values}.
     *
     * @throws IllegalArgumentException If the values vector has a different length than the number of rows of this matrix.
     * @throws
     */
    @Override
    public CooMatrix setRow(CooVector values, int rowIndex) {
        return RealSparseMatrixGetSet.setRow(this, rowIndex, values);
    }


    /**
     * Sets a specified row of this matrix to an array.
     *
     * @param row Array containing values to replace specified row in this matrix.
     * @param rowIdx Index of the row to set.
     *
     * @return If this matrix is dense, the row set operation is done in place and a reference to this matrix is returned.
     * If this matrix is sparse a copy will be created with the new row and returned.
     */
    public CooMatrix setRow(double[] row, int rowIdx) {
        return RealSparseMatrixGetSet.setRow(this, rowIdx, row);
    }


    /**
     *  Converts this real sparse COO matrix to an equivalent complex sparse COO matrix.
     * @return A complex sparse COO matrix equivalent to this matrix.
     */
    public CooCMatrix toComplex() {
        return new CooCMatrix(shape, entries, rowIndices.clone(), colIndices.clone());
    }


    /**
     * Computes the element-wise product between two matrices.
     * @param b Second matrix in the element-wise product.
     * @return The element-wise product of this matrix and {@code b}.
     */
    public CooCMatrix elemMult(CooCMatrix b) {
        return RealComplexSparseMatrixOperations.elemMult(b, this);
    }


    /**
     * Computes the element-wise product between two matrices.
     * @param b Second matrix in the element-wise product.
     * @return The element-wise product of this matrix and {@code b}.
     */
    public CooMatrix elemMult(Matrix b) {
        return RealDenseSparseMatrixOperations.elemMult(b, this);
    }


    /**
     * Computes the element-wise product between two matrices.
     * @param b Second matrix in the element-wise product.
     * @return The element-wise product of this matrix and {@code b}.
     */
    public CooCMatrix elemMult(CMatrix b) {
        return RealComplexDenseSparseMatrixOperations.elemMult(b, this);
    }


    /**
     * Checks if an object is equal to this matrix.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link CooMatrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooMatrix src2 = (CooMatrix) object;

        return RealSparseEquals.cooMatrixEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < entries.length; i++) {
            if (entries[i] != 0.0) {
                result = 31*result + Double.hashCode(entries[i]);
                result = 31*result + Integer.hashCode(rowIndices[i]);
                result = 31*result + Integer.hashCode(colIndices[i]);
            }
        }

        return result;
    }


    /**
     * Formats this sparse matrix as a human-readable string.
     * @return A human-readable string representing this sparse matrix.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("Non-zero entries: [");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        if(entries.length > 0) {
            // Get entries up until the stopping point.
            for(int i=0; i<stopIndex; i++) {
                value = StringUtils.ValueOfRound(entries[i], PrintOptions.getPrecision());
                width = PrintOptions.getPadding() + value.length();
                value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            if(stopIndex < size-1) {
                width = PrintOptions.getPadding() + 3;
                value = "...";
                value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            // Get last entry now
            value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");

        result.append("Row Indices: ").append(Arrays.toString(rowIndices)).append("\n");
        result.append("Column Indices: ").append(Arrays.toString(colIndices));

        return result.toString();
    }
}
