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

import org.flag4j.algebraic_structures.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCooSemiringVector;
import org.flag4j.arrays.dense.SemiringMatrix;
import org.flag4j.arrays.dense.SemiringVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.field_ops.CooFieldEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;

import java.util.List;
import java.util.function.BinaryOperator;

/**
 * Represents a sparse vector whose non-zero elements are stored in Coordinate List (COO) format, with all data elements
 * belonging to a specified {@link Semiring} type.
 *
 * <p>The COO format stores sparse vector data as a list of coordinates (indices) coupled with their
 * corresponding non-zero values, rather than allocating memory for every element in the full vector shape. This
 * allows efficient representation and manipulation of large vector containing a substantial number of zeros.
 *
 * <p>A sparse COO vector is stored as:
 * <ul>
 *     <li><b>Shape:</b> The full {@link #shape}/{@link #size} of the vector specifying the total number of values (including zeros)
 *     in the vector.</li>
 *
 *     <li><b>Data:</b> Non-zero values are stored in a one-dimensional array, {@link #data}. Any element not specified in
 *     {@code data} is implicitly zero. It is also possible to explicitly store zero values in this array, although this
 *     is generally not desirable. To remove explicitly defined zeros, use {@link #dropZeros()}.</li>
 *
 *     <li><b>Indices:</b> Non-zero values are associated with their coordinates in the vector via a single 1D array:
 *     {@link #indices}. This array specifies the positions of each
 *     non-zero entry in {@link #data}. The total number of non-zero elements is given by {@link #nnz}.</li>
 * </ul>
 *
 * <p>The total number of non-zero elements ({@link #nnz}) and the shape/size is fixed for a given instance, but the values
 * in {@link #data} and their corresponding {@link #indices} may be updated. Many operations
 * assume that the indices are sorted lexicographically, but this is not strictly enforced.
 * All provided operations preserve the lexicographical sorting of indices. If there is any doubt about the ordering of
 * indices, use {@link #sortIndices()} to ensure they are sorted. COO tensors may also store multiple entries
 * for the same index (referred to as an uncoalesced tensor). To combine all duplicated entries use {@link #coalesce()} or
 * {@link #coalesce(BinaryOperator)}.
 *
 * <p>COO vectors are optimized for "hyper-sparse" scenarios where the proportion of non-zero elements is extremely low,
 * offering significant memory savings and potentially more efficient computational operations than equivalent dense
 * representations.
 *
 * <h3>Example Usage:</h3>
 * <pre>{@code
 * // shape, data, and indices for COO vector.
 * Shape shape = new Shape(512);
 * BoolSemiring[] data = {
 *      new BoolSemiring(true),  new BoolSemiring(false), new BoolSemiring(false),
 *      new BoolSemiring(false), new BoolSemiring(true),  new BoolSemiring(true)
 * };
 * int[] indices = {0, 4, 128, 128, 128, 256};
 *
 * // Create COO vector.
 * CooSemiringVector<BoolSemiring> vector = new CooSemiringVector(shape, data, indices);
 *
 * // Sum vectors.
 * CooSemiringVector<BoolSemiring> sum = vector.add(vector);
 *
 * // Compute vector inner product.
 * BoolSemiring prod = vector.inner(vector);
 *
 * // Compute vector outer product.
 * SemiringMatrix<BoolSemiring> prod = vector.outer(vector);
 * }</pre>
 *
 * @param <T> The type of elements stored in this vector, constrained by the {@link Semiring} interface.
 * @see CooSemiringTensor
 * @see CooSemiringMatrix
 * @see SemiringMatrix
 * @see Semiring
 */
public class CooSemiringVector<T extends Semiring<T>> extends AbstractCooSemiringVector<
        CooSemiringVector<T>, SemiringVector<T>, CooSemiringMatrix<T>, SemiringMatrix<T>, T> {

    private static final long serialVersionUID = 1L;


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param size
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     * @param indices
     */
    public CooSemiringVector(int size, T[] entries, int[] indices) {
        super(new Shape(size), entries, indices);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param size
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     * @param indices
     */
    public CooSemiringVector(Shape shape, T[] entries, int[] indices) {
        super(shape, entries, indices);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param Shape The shape of this vector. Must be rank-1.
     * @param entries The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooSemiringVector(Shape shape, List<T> entries, List<Integer> indices) {
        super(shape, (T[]) entries.toArray(Semiring[]::new), ArrayUtils.fromIntegerList(indices));
    }


    /**
     * Creates a zero vector of the specified {@code size}.
     */
    public CooSemiringVector(int size) {
        super(new Shape(size), (T[]) new Semiring[0], new int[0]);
    }


    /**
     * Creates sparse COO vector with the specified {@code size}, non-zero data, and non-zero indices.
     *
     * @param size The size of this vector.
     * @param entries The non-zero data of this vector.
     * @param indices The indices of the non-zero values.
     */
    public CooSemiringVector(int size, List<T> entries, List<Integer> indices) {
        super(new Shape(size), (T[]) entries.toArray(Semiring[]::new), ArrayUtils.fromIntegerList(indices));
    }


    /**
     * Constructs a sparse COO vector of the same type as this vector with the specified non-zero data and indices.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Non-zero data of the vector to construct.
     * @param indices Non-zero row indices of the vector to construct.
     *
     * @return A sparse COO vector of the same type as this vector with the specified non-zero data and indices.
     */
    @Override
    public CooSemiringVector<T> makeLikeTensor(Shape shape, T[] entries, int[] indices) {
        return new CooSemiringVector<>(shape, entries, indices);
    }


    /**
     * Constructs a dense vector of a similar type as this vector with the specified shape and data.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Entries of the vector to construct.
     *
     * @return A dense vector of a similar type as this vector with the specified data.
     */
    @Override
    public SemiringVector<T> makeLikeDenseTensor(Shape shape, T... entries) {
        return new SemiringVector<>(shape, entries);
    }


    /**
     * Constructs a dense matrix of a similar type as this vector with the specified shape and data.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A dense matrix of a similar type as this vector with the specified data.
     */
    @Override
    public SemiringMatrix<T> makeLikeDenseMatrix(Shape shape, T... entries) {
        return new SemiringMatrix<>(shape, entries);
    }


    /**
     * Constructs a COO vector with the specified shape, non-zero data, and non-zero indices.
     *
     * @param shape Shape of the vector.
     * @param entries Non-zero values of the vector.
     * @param indices Indices of the non-zero values in the vector.
     *
     * @return A COO vector of the same type as this vector with the specified shape, non-zero data, and non-zero indices.
     */
    @Override
    public CooSemiringVector<T> makeLikeTensor(Shape shape, List<T> entries, List<Integer> indices) {
        return new CooSemiringVector<>(shape, entries, indices);
    }


    /**
     * Constructs a COO matrix with the specified shape, non-zero data, and row and column indices.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Non-zero data of the matrix.
     * @param rowIndices Row indices of the matrix.
     * @param colIndices Column indices of the matrix.
     *
     * @return A COO matrix of similar type as this vector with the specified shape, non-zero data, and non-zero row/col indices.
     */
    @Override
    public CooSemiringMatrix<T> makeLikeMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        return new CooSemiringMatrix<>(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public CooSemiringVector<T> makeLikeTensor(Shape shape, T[] entries) {
        return new CooSemiringVector<>(shape, entries, indices.clone());
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link CooFieldVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooSemiringVector<T> src2 = (CooSemiringVector<T>) object;

        return CooFieldEquals.cooVectorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < data.length; i++) {
            if (!data[i].isZero()) {
                result = 31*result + data[i].hashCode();
                result = 31*result + Integer.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     * <p>If {@code true}, the vector will be converted to a matrix representing a column vector.
     * <p>If {@code false}, The vector will be converted to a matrix representing a row vector.
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public CooSemiringMatrix<T> toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[data.length];

            return new CooSemiringMatrix<T>(size, 1, data.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[data.length];
            int[] colIndices = indices.clone();

            return new CooSemiringMatrix<T>(1, size, data.clone(), rowIndices, colIndices);
        }
    }


    /**
     * Converts this matrix to an equivalent rank 1 tensor.
     *
     * @return A tensor which is equivalent to this matrix.
     */
    @Override
    public CooSemiringTensor<T> toTensor() {
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(new int[][]{indices});
        return new CooSemiringTensor(shape, data.clone(), tIndices);
    }


    /**
     * Converts this vector to an equivalent tensor with the specified shape.
     *
     * @param newShape New shape for the tensor. Can be any rank but must be broadcastable to {@link #shape this.shape}.
     *
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooSemiringTensor<T> toTensor(Shape newShape) {
        return toTensor().reshape(newShape);
    }


    /**
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flatten data of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("nnz: ").append(nnz).append("\n");
        result.append("Non-zero data: [");

        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        boolean centering = PrintOptions.useCentering();
        int precision = PrintOptions.getPrecision();

        if(size > 0) {
            int stopIndex = Math.min(maxCols -1, size-1);
            int width;
            String value;

            // Get data up until the stopping point.
            for(int i = 0; i<stopIndex; i++) {
                value = StringUtils.ValueOfRound(data[i], precision);
                width = padding + value.length();
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            if(stopIndex < size-1) {
                width = padding + 3;
                value = "...";
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            // Get last entry now
            value = StringUtils.ValueOfRound(data[size-1], precision);
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");
        result.append("Indices: ")
                .append(PrettyPrint.abbreviatedArray(indices, maxCols, padding, centering));

        return result.toString();
    }
}
