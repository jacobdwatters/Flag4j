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
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.CooFieldMatrixBase;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.io.PrintOptions;
import org.flag4j.operations.dense_sparse.coo.field_ops.DenseCooFieldMatrixOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseMatrixOperations;
import org.flag4j.operations.sparse.coo.field_ops.CooFieldEquals;
import org.flag4j.operations.sparse.coo.field_ops.CooFieldMatMult;
import org.flag4j.operations.sparse.coo.field_ops.CooFieldMatrixGetSet;
import org.flag4j.operations.sparse.coo.real_complex.RealComplexSparseMatrixOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.List;

/**
 * <p>A complex sparse matrix stored in coordinate list (COO) format. The {@link #entries} of this COO tensor are
 * {@link Complex128}'s.</p>
 *
 * <p>The {@link #entries non-zero entries} and non-zero indices of a COO matrix are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.</p>
 *
 * <p>Sparse matrices allow for the efficient storage of and operations on matrices that contain many zero values.</p>
 *
 * <p>COO matrices are optimized for hyper-sparse matrices (i.e. matrices which contain almost all zeros relative to the size of the
 * matrix).</p>
 *
 * <p>A sparse COO matrix is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #entries} of the matrix. All other entries in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #rowIndices row indices} of the non-zero values in the sparse matrix.</li>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the COO matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified but any operations implemented in this
 * class will preserve the lexicographical sorting.</p>
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.</p>
 */
public class CooCMatrix extends CooFieldMatrixBase<CooCMatrix, CMatrix, CooCVector, CVector, Complex128> {

    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(Shape shape, Field<Complex128>[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries, rowIndices, colIndices);
        ValidateParameters.ensureRank(shape, 2);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(Shape shape, List<Field<Complex128>> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(shape, entries, rowIndices, colIndices);
        ValidateParameters.ensureRank(shape, 2);
        if(entries.size() == 0 || entries.get(0) == null)
            setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param shape The shape of the matrix.
     */
    public CooCMatrix(Shape shape) {
        super(shape, new Complex128[0], new int[0], new int[0]);
        ValidateParameters.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(int rows, int cols, Field<Complex128>[] entries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), entries, rowIndices, colIndices);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(int rows, int cols, List<Field<Complex128>> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(new Shape(rows, cols), entries, rowIndices, colIndices);
        if(super.entries.length == 0 || super.entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    public CooCMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new Complex128[0], new int[0], new int[0]);
        ValidateParameters.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero entries, non-zero indices, and shape.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries Non-zero entries of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(Shape shape, double[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, ArrayUtils.wrapAsComplex128(entries, null), rowIndices, colIndices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a square sparse COO matrix with the specified {@code size} filled with zeros.
     * @param size Size of the square matrix to construct.
     */
    public CooCMatrix(int size) {
        super(new Shape(size, size), new Complex128[0], new int[0], new int[0]);
    }


    /**
     * Constructs a square complex sparse COO matrix with the specified {@code size}, non-zero values, and non-zero indices.
     * @param size Size of the square matrix to construct.
     * @param entries Non-zero entries of the sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(int size, Field<Complex128>[] entries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size), entries, rowIndices, colIndices);
    }


    /**
     * Constructs a copy of the specified complex sparse COO matrix.
     * @param b Matrix to copy.
     */
    public CooCMatrix(CooCMatrix b) {
        super(b.shape, b.entries.clone(), b.rowIndices.clone(), b.colIndices.clone());
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
    public CooCMatrix set(Complex128 value, int... indices) {
        ValidateParameters.ensureValidIndex(shape, indices);
        return (CooCMatrix) CooFieldMatrixGetSet.matrixSet(this, indices[0], indices[1], value);
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
    public CooCMatrix set(double value, int... indices) {
        return set(new Complex128(value), indices);
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
    public CooCMatrix makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CooCMatrix(shape, entries, rowIndices.clone(), colIndices.clone());
    }


    /**
     * Constructs a COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Row indices of the non-zero values in the matrix.
     * @param colIndices Column indices of the non-zero values in the matrix.
     *
     * @return A COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    @Override
    public CooCMatrix makeLikeTensor(Shape shape, Field<Complex128>[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Row indices of the non-zero values in the matrix.
     * @param colIndices Column indices of the non-zero values in the matrix.
     *
     * @return A COO field matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    @Override
    public CooCMatrix makeLikeTensor(Shape shape, List<Field<Complex128>> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a dense field matrix which with the specified {@code shape} and {@code entries}.
     *
     * @param shape Shape of the matrix.
     * @param entries Entries of the dense matrix/.
     *
     * @return A dense field matrix with the specified {@code shape} and {@code entries}.
     */
    @Override
    public CMatrix makeDenseTensor(Shape shape, Field<Complex128>[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Constructs a vector of similar type to this matrix.
     *
     * @param size The size of the vector.
     * @param entries The non-zero entries of the vector.
     * @param indices The indices of the non-zero values of the vector.
     *
     * @return A vector of similar type to this matrix with the specified size, non-zero entries, and indices.
     */
    @Override
    public CooCVector makeLikeVector(int size, Field<Complex128>[] entries, int[] indices) {
        return new CooCVector(size, entries, indices);
    }


    /**
     * Constructs a vector of similar type to this matrix.
     *
     * @param size The size of the vector.
     * @param entries The non-zero entries of the vector.
     * @param indices The indices of the non-zero values of the vector.
     *
     * @return A vector of similar type to this matrix with the specified size, non-zero entries, and indices.
     */
    @Override
    public CooCVector makeLikeVector(int size, List<Field<Complex128>> entries, List<Integer> indices) {
        return new CooCVector(size, entries, indices);
    }


    /**
     * Converts this sparse COO matrix to an equivalent sparse CSR matrix.
     *
     * @return A sparse CSR matrix equivalent to this sparse COO matrix.
     */
    @Override
    public CsrCMatrix toCsr() {
        int[] rowPointers = new int[numRows + 1];

        // Count number of entries per row.
        for(int i=0; i<nnz; i++)
            rowPointers[rowIndices[i] + 1]++;

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<numRows; i++)
            rowPointers[i+1] += rowPointers[i];

        return new CsrCMatrix(shape, entries.clone(), rowPointers, colIndices.clone());
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
    public CVector mult(CooCVector b) {
        return new CVector(CooFieldMatMult.standardVector(
                entries, rowIndices, colIndices, shape, b.entries, b.indices));
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    public CooCMatrix augment(CooVector b) {
        ValidateParameters.ensureEquals(numRows, b.size);

        Shape destShape = new Shape(numRows, numCols + 1);
        Complex128[] destEntries = new Complex128[nnz + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries and indices from this matrix.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        System.arraycopy(rowIndices, 0, destRowIndices, 0, entries.length);
        System.arraycopy(colIndices, 0, destColIndices, 0, entries.length);

        // Copy entries and indices from vector.
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);
        System.arraycopy(b.indices, 0, destRowIndices, entries.length, b.entries.length);
        Arrays.fill(destColIndices, entries.length, destColIndices.length, numCols);

        CooCMatrix dest = new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure that the indices are sorted properly.

        return dest;
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(MatrixMixin, int)
     */
    public CooCMatrix augment(CooMatrix b) {
        ValidateParameters.ensureEquals(numRows, b.numRows);

        Shape destShape = new Shape(numRows, numCols + b.numCols);
        Complex128[] destEntries = new Complex128[entries.length + b.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        System.arraycopy(entries, 0, destEntries, 0, entries.length);
        ArrayUtils.arraycopy(b.entries, 0, destEntries, entries.length, b.entries.length);

        // Copy row indices.
        System.arraycopy(rowIndices, 0, destRowIndices, 0, rowIndices.length);
        System.arraycopy(b.rowIndices, 0, destRowIndices, rowIndices.length, b.rowIndices.length);

        // Copy column indices (with shifts if appropriate).
        int[] shifted = b.colIndices.clone();
        System.arraycopy(colIndices, 0, destColIndices, 0, colIndices.length);
        System.arraycopy(ArrayUtils.shift(numCols, shifted), 0,
                destColIndices, colIndices.length, b.colIndices.length);

        CooCMatrix dest = new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure indices are sorted properly.

        return dest;
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with the scalar {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCMatrix sub(CooMatrix b) {
        return RealComplexSparseMatrixOperations.sub(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     * @return The difference of this tensor with the scalar {@code b}.
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCMatrix elemMult(CooMatrix b) {
        return RealComplexSparseMatrixOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     * @return The difference of this tensor with the scalar {@code b}.
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCMatrix elemMult(Matrix b) {
        return RealComplexDenseSparseMatrixOperations.elemMult(b, this);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     * @return The difference of this tensor with the scalar {@code b}.
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCMatrix elemMult(CMatrix b) {
        return (CooCMatrix) DenseCooFieldMatrixOperations.elemMult(b, this);
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this vector.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link Matrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        // Quick returns if possible.
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        return CooFieldEquals.cooMatrixEquals(this, (CooCMatrix) object);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < entries.length; i++) {
            if (!entries[i].isZero()) {
                result = 31*result + entries[i].hashCode();
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
                value = StringUtils.ValueOfRound((Complex128) entries[i], PrintOptions.getPrecision());
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
            value = StringUtils.ValueOfRound((Complex128) entries[size-1], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");

        result.append("Row Indices: ").append(Arrays.toString(rowIndices)).append("\n");
        result.append("Col Indices: ").append(Arrays.toString(colIndices));

        return result.toString();
    }
}
