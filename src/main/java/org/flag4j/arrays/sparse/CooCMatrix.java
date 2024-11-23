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
import org.flag4j.arrays.backend.AbstractTensor;
import org.flag4j.arrays.backend.field.AbstractCooFieldMatrix;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.dense.real.RealDenseTranspose;
import org.flag4j.linalg.operations.sparse.coo.field_ops.CooFieldEquals;
import org.flag4j.linalg.operations.sparse.coo.semiring_ops.CooSemiringMatMult;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;
import java.util.List;


/**
 * <p>A complex sparse matrix stored in coordinate list (COO) format. The {@link #entries} of this COO tensor are
 * primitive doubles.
 *
 * <p>The {@link #entries non-zero entries} and non-zero indices of a COO matrix are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.
 *
 * <p>COO matrices are well-suited for incremental matrix construction and modification but may not have ideal efficiency for matrix
 * operations like matrix multiplication. For heavy computations, it may be better to construct a matrix as a {@code CooMatrix} then
 * convert to a {@link CsrCMatrix} (using {@link #toCsr()}) as CSR (compressed sparse row) matrices are generally better suited for
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
 * <p>Note: many operations assume that the entries of the COO matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified. Any operations implemented in this
 * class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.
 */
public class CooCMatrix extends AbstractCooFieldMatrix<CooCMatrix, CMatrix, CooCVector, Complex128> {

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
        super(shape,
                entries.toArray(new Complex128[entries.size()]),
                ArrayUtils.fromIntegerList(rowIndices),
                ArrayUtils.fromIntegerList(colIndices));
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
        super(new Shape(rows, cols),
                entries.toArray(new Complex128[entries.size()]),
                ArrayUtils.fromIntegerList(rowIndices),
                ArrayUtils.fromIntegerList(colIndices));
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
     * Constructs a sparse COO tensor of the same type as this tensor with the specified non-zero entries and indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero entries of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     *
     * @return A sparse COO tensor of the same type as this tensor with the specified non-zero entries and indices.
     */
    @Override
    public CooCMatrix makeLikeTensor(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a COO matrix with the specified shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     *
     * @return A COO matrix with the specified shape, non-zero entries, and non-zero indices.
     */
    @Override
    public CooCMatrix makeLikeTensor(Shape shape, List<Field<Complex128>> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a sparse COO vector of a similar type to this COO matrix.
     *
     * @param shape Shape of the vector. Must be rank 1.
     * @param entries Non-zero entries of the COO vector.
     * @param indices Non-zero indices of the COO vector.
     *
     * @return A sparse COO vector of a similar type to this COO matrix.
     */
    @Override
    public CooCVector makeLikeVector(Shape shape, Field<Complex128>[] entries, int[] indices) {
        return new CooCVector(shape.totalEntriesIntValueExact(), entries, indices);
    }


    /**
     * Constructs a dense tensor with the specified {@code shape} and {@code entries} which is a similar type to this sparse tensor.
     *
     * @param shape Shape of the dense tensor.
     * @param entries Entries of the dense tensor.
     *
     * @return A dense tensor with the specified {@code shape} and {@code entries} which is a similar type to this sparse tensor.
     */
    @Override
    public CMatrix makeLikeDenseTensor(Shape shape, Field<Complex128>[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code entries}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code entries}.
     */
    @Override
    public CooCMatrix makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CooCMatrix(shape, entries, rowIndices.clone(), colIndices.clone());
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 Tensor to contract with this tensor.
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
    public AbstractTensor<?, Field<Complex128>[], Complex128> tensorDot(CooCMatrix src2, int[] aAxes, int[] bAxes) {
        return toTensor().tensorDot(src2.toTensor(), aAxes, bAxes);
    }


    /**
     * Constructs a sparse CSR matrix of a similar type to this sparse COO matrix.
     *
     * @param shape Shape of the CSR matrix to construct.
     * @param entries Non-zero entries of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A CSR matrix of a similar type to this sparse COO matrix.
     */
    @Override
    public CsrCMatrix makeLikeCsrMatrix(Shape shape, Field<Complex128>[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor which is equivalent to this matrix.
     */
    @Override
    public CooCTensor toTensor() {
        int[][] indices = {rowIndices.clone(), colIndices.clone()};
        indices = RealDenseTranspose.blockedIntMatrix(indices);
        return new CooCTensor(shape, entries.clone(), indices);
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified shape.
     *
     * @param newShape New shape for the tensor. Can be any rank but must be broadcastable to {@link #shape this.shape}.
     *
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooCTensor toTensor(Shape newShape) {
        return toTensor().reshape(newShape);
    }


    /**
     * Computes the matrix-vector multiplication of a vector with this matrix.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of multiplying this matrix with {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the size of
     *                                {@code b}.
     */
    @Override
    public CVector mult(CooCVector b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[b.size];
        CooSemiringMatMult.standardVector(
                entries, rowIndices, colIndices, shape,
                b.entries, b.indices, dest);

        return new CVector(dest);
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link CooCMatrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
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
