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
import org.flag4j.arrays.backend.field.AbstractCooFieldVector;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.common.complex.Complex128Ops;
import org.flag4j.linalg.operations.dense.real.RealDenseTranspose;
import org.flag4j.linalg.operations.sparse.coo.field_ops.CooFieldEquals;
import org.flag4j.linalg.operations.sparse.coo.real_complex.RealComplexSparseVectorOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.List;


/**
 * <p>A complex sparse vector stored in coordinate list (COO) format. The {@link #entries} of this COO vector are
 * {@link Complex128}'s.</p>
 *
 * <p>The {@link #entries non-zero entries} and {@link #indices non-zero indices} of a COO vector are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.</p>
 *
 * <p>Sparse vectors allow for the efficient storage of and operations on vectors that contain many zero values.</p>
 *
 * <p>COO vectors are optimized for hyper-sparse vectors (i.e. vectors which contain almost all zeros relative to the size of the
 * vector).</p>
 *
 * <p>A sparse COO vector is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape}/{@link #size} of the vector.</li>
 *     <li>The non-zero {@link #entries} of the vector. All other entries in the vector are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #indices} of the non-zero values in the sparse vector.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the COO vector are sorted lexicographically. However, this is not explicitly
 * verified. Every operation implemented in this class will preserve the lexicographical sorting.</p>
 *
 * <p>If indices need to be sorted for any reason, call {@link #sortIndices()}.</p>
 */
public class CooCVector extends AbstractCooFieldVector<CooCVector, CVector, CooCMatrix, CMatrix, Complex128> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param size Full size of the vector.
     * @param entries Non-zero entries of the sparse vector.
     * @param indices Non-zero indices of the sparse vector.
     */
    public CooCVector(int size, Field<Complex128>[] entries, int[] indices) {
        super(new Shape(size), entries, indices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Full shape of the vector. Must be rank 1.
     * @param entries Non-zero entries of the sparse vector.
     * @param indices Non-zero indices of the sparse vector.
     */
    public CooCVector(Shape shape, Field<Complex128>[] entries, int[] indices) {
        super(shape, entries, indices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a complex COO vector with the specified size, non-zero entries, and non-zero indices.
     * @param size Full size of the vector.
     * @param entries The non-zero entries of the vector.
     * @param indices The indices of the non-zero entries.
     */
    public CooCVector(int size, List<Field<Complex128>> entries, List<Integer> indices) {
        super(new Shape(size), entries.toArray(new Complex128[0]), ArrayUtils.fromIntegerList(indices));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a complex COO vector with the specified size, non-zero entries, and non-zero indices.
     * @param shape Full shape of the sparse vector. Must be rank 1.
     * @param entries The non-zero entries of the vector.
     * @param indices The indices of the non-zero entries.
     */
    public CooCVector(Shape shape, List<Field<Complex128>> entries, List<Integer> indices) {
        super(shape, entries.toArray(new Complex128[0]), ArrayUtils.fromIntegerList(indices));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero vector of the specified {@code size}.
     * @param size Full size of the vector.
     */
    public CooCVector(int size) {
        super(new Shape(size), new Complex128[0], new int[0]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a sparse complex COO vector from an array of double values.
     * @param size Full size of the vector.
     * @param entries Non-zero entries of the sparse vector.
     * @param indices Non-zero indices of the sparse vector.
     */
    public CooCVector(int size, double[] entries, int[] indices) {
        super(new Shape(size), ArrayUtils.wrapAsComplex128(entries, null), indices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a copy of the specified vector.
     * @param b The vector to create a copy of.
     */
    public CooCVector(CooCVector b) {
        super(b.shape, b.entries.clone(), b.indices.clone());
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a sparse COO vector of the same type as this vector with the specified non-zero entries and indices.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Non-zero entries of the vector to construct.
     * @param indices Non-zero row indices of the vector to construct.
     *
     * @return A sparse COO vector of the same type as this vector with the specified non-zero entries and indices.
     */
    @Override
    public CooCVector makeLikeTensor(Shape shape, Field<Complex128>[] entries, int[] indices) {
        return new CooCVector(shape, entries, indices);
    }


    /**
     * Constructs a dense vector of a similar type as this vector with the specified shape and entries.
     *
     * @param shape Shape of the vector to construct.
     * @param entries Entries of the vector to construct.
     *
     * @return A dense vector of a similar type as this vector with the specified entries.
     */
    @Override
    public CVector makeLikeDenseTensor(Shape shape, Field<Complex128>... entries) {
        ValidateParameters.ensureRank(shape, 1);
        return new CVector(entries);
    }


    /**
     * Constructs a dense matrix of a similar type as this vector with the specified shape and entries.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A dense matrix of a similar type as this vector with the specified entries.
     */
    @Override
    public CMatrix makeLikeDenseMatrix(Shape shape, Field<Complex128>... entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Constructs a COO vector with the specified shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the vector.
     * @param entries Non-zero values of the vector.
     * @param indices Indices of the non-zero values in the vector.
     *
     * @return A COO vector of the same type as this vector with the specified shape, non-zero entries, and non-zero indices.
     */
    @Override
    public CooCVector makeLikeTensor(Shape shape, List<Field<Complex128>> entries, List<Integer> indices) {
        return new CooCVector(shape, entries, indices);
    }


    /**
     * Constructs a COO matrix with the specified shape, non-zero entries, and row and column indices.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Non-zero entries of the matrix.
     * @param rowIndices Row indices of the matrix.
     * @param colIndices Column indices of the matrix.
     *
     * @return A COO matrix of similar type as this vector with the specified shape, non-zero entries, and non-zero row/col indices.
     */
    @Override
    public CooCMatrix makeLikeMatrix(Shape shape, Field<Complex128>[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
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
    public CooCVector makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CooCVector(shape, entries, indices.clone());
    }


    /**
     * Converts this matrix to an equivalent rank 1 tensor.
     *
     * @return A tensor which is equivalent to this matrix.
     */
    @Override
    public CooCTensor toTensor() {
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(new int[][]{indices.clone()});
        return new CooCTensor(shape, entries.clone(), tIndices);
    }


    /**
     * Converts this vector to an equivalent tensor with the specified shape.
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
     * Converts this complex vector to a real vector.
     * @return A real vector containing the real components of all non-zero values in this vector. The imaginary components are
     * ignored.
     */
    public CooVector toReal() {
        return new CooVector(size, Complex128Ops.toReal(entries), indices.clone());
    }


    /**
     * Computes the element-wise sum of two vectors.
     * @param b Second vector in the sum.
     * @return The element-wise sum of this vector and {@code b}.
     */
    public CooCVector add(CooVector b) {
        return RealComplexSparseVectorOperations.add(this, b);
    }


    /**
     * Normalizes this vector to a unit length vector.
     *
     * @return This vector normalized to a unit length.
     */
    @Override
    public CooCVector normalize() {
        return div(magAsDouble());
    }


    /**
     * Computes the magnitude of this vector.
     *
     * @return The magnitude of this vector.
     */
    @Override
    public Complex128 mag() {
        return new Complex128(magAsDouble());
    }


    /**
     * Computes the magnitude of this vector as a double value.
     * @return The magnitude of this vector as a double value.
     */
    public double magAsDouble() {
        double mag = 0;

        for(int i = 0, size=nnz; i < size; i++) {
            Complex128 v = (Complex128) entries[i];
            mag += (v.re*v.re + v.im*v.im);
        }

        return Math.sqrt(mag);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link CooCVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        return CooFieldEquals.cooVectorEquals(this, (CooCVector) object);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < entries.length; i++) {
            if (!entries[i].isZero()) {
                result = 31*result + ((Complex128) entries[i]).hashCode();
                result = 31*result + Integer.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flatten entries of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("Non-zero entries: [");

        if(size > 0) {
            int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
            int width;
            String value;

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
        result.append("Indices: ").append(Arrays.toString(indices));

        return result.toString();
    }
}
