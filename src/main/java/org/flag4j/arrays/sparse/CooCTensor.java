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
import org.flag4j.algebraic_structures.fields.Complex64;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.CooFieldTensorBase;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.dense.real.RealDenseTranspose;
import org.flag4j.linalg.operations.sparse.coo.field_ops.CooFieldEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.List;


/**
 * <p>Sparse complex tensor stored in coordinate list (COO) format. The entries of this COO tensor are of type {@link Complex128}</p>
 *
 * <p>The non-zero entries and non-zero indices of a COO tensor are mutable but the {@link #shape} and total number of
 * {@link #entries non-zero entries} is fixed.</p>
 *
 * <p>Sparse tensors allow for the efficient storage of and operations on tensors that contain many zero values.</p>
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e. tensors which contain almost all zeros relative to the size of the
 * tensor).</p>
 *
 * <p>A sparse COO tensor is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #entries} of the tensor. All other entries in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #entries}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many operations assume indices to be sorted in a
 *     row-major format (i.e. last index increased fastest) but often this is not explicitly verified.</p>
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero entries in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the ND
 *     index of {@code entries[i]}.</p>
 *     </li>
 * </ul>
 */
public class CooCTensor extends CooFieldTensorBase<CooCTensor, CTensor, Complex128> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor.
     * @param indices Indices of the non-zero entries of this tensor.
     */
    public CooCTensor(Shape shape, Field<Complex128>[] entries, int[][] indices) {
        super(shape, entries, indices);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor.
     * @param indices Indices of the non-zero entries of this tensor.
     */
    public CooCTensor(Shape shape, List<Field<Complex128>> entries, List<int[]> indices) {
        super(shape, entries.toArray(new Complex128[0]), indices.toArray(new int[0][]));
        if(super.entries.length == 0 || super.entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor.
     * @param indices Indices of the non-zero entries of this tensor.
     */
    public CooCTensor(Shape shape, Complex64[] entries, int[][] indices) {
        super(shape, new Complex128[entries.length], indices);
        setZeroElement(Complex128.ZERO);

        for(int i=0, size=entries.length; i<size; i++)
            this.entries[i] = new Complex128((Complex64) entries[i]);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor.
     * @param indices Indices of the non-zero entries of this tensor.
     */
    public CooCTensor(Shape shape) {
        super(shape, new Complex128[0], new int[0][shape.getRank()]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor.
     * @param indices Indices of the non-zero entries of this tensor.
     */
    public CooCTensor(Shape shape, double[] entries, int[][] indices) {
        super(shape, ArrayUtils.wrapAsComplex128(entries, null), indices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a copy of a sparse COO tensor.
     * @param b Tensor to construct copy of.
     */
    public CooCTensor(CooCTensor b) {
        super(b.shape, b.entries.clone(), ArrayUtils.deepCopy(b.indices, null));
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param index Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public CooCTensor set(Complex128 value, int... index) {
        ValidateParameters.ensureValidIndex(shape, index);
        CooCTensor dest;

        // Check if value already exists in tensor.
        int idx = -1;
        for(int i=0; i<indices.length; i++) {
            if(Arrays.equals(indices[i], index)) {
                idx = i;
                break; // Found in tensor, no need to continue.
            }
        }

        if(idx > -1) {
            // Copy entries and set new value.
            dest = new CooCTensor(shape, entries.clone(), ArrayUtils.deepCopy(indices, null));
            dest.entries[idx] = value;
            dest.indices[idx] = index;
        } else {
            // Copy old indices and insert new one.
            int[][] newIndices = new int[indices.length + 1][getRank()];
            ArrayUtils.deepCopy(indices, newIndices);
            newIndices[indices.length] = index;

            // Copy old entries and insert new one.
            Field<Complex128>[] newEntries = Arrays.copyOf(entries, entries.length+1);
            newEntries[newEntries.length-1] = value;

            dest = new CooCTensor(shape, newEntries, newIndices);
            dest.sortIndices();
        }

        return dest;
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and entries.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Entries of the spares tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and entries.
     */
    @Override
    public CooCTensor makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CooCTensor(shape, entries, indices.clone());
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given the shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero entries of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CooCTensor makeLikeTensor(Shape shape, Field<Complex128>[] entries, int[][] indices) {
        return new CooCTensor(shape, entries, indices);
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given the shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero entries of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CooCTensor makeLikeTensor(Shape shape, List<Field<Complex128>> entries, List<int[]> indices) {
        return new CooCTensor(shape, entries.toArray(new Complex128[0]), indices.toArray(new int[0][]));
    }


    /**
     * Makes a dense tensor with the specified shape and entries which is a similar type to this sparse tensor.
     *
     * @param shape Shape of the dense tensor.
     * @param entries Entries of the dense tensor.
     *
     * @return A dense tensor with the specified shape and entries which is a similar type to this sparse tensor.
     */
    @Override
    public CTensor makeDenseTensor(Shape shape, Field<Complex128>[] entries) {
        return new CTensor(shape, entries);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor equivalent to this sparse tensor.
     */
    @Override
    public CTensor toDense() {
        Complex128[] entries = new Complex128[totalEntries().intValueExact()];
        for(int i=0; i<nnz; i++)
            entries[shape.entriesIndex(indices[i])] = (Complex128) this.entries[i];

        return new CTensor(shape, entries);
    }


    /**
     * Converts this tensor to a matrix with specified shape.
     * @param newShape Shape of matrix to convert this tensor to. Shape must be broadcastable with this tensors shape and have rank 2.
     * @return A matrix of the specified shape with the same non-zero entries as this tensor.
     */
    public CooCMatrix toMatrix(Shape newShape) {
        ValidateParameters.ensureRank(newShape, 2);
        CooCTensor t = reshape(newShape); // Reshape as rank 2 tensor. Broadcastable check made here.
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(t.indices);

        return new CooCMatrix(newShape, t.entries.clone(), tIndices[0], tIndices[1]);
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param index Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    public CooCTensor set(double value, int... indices) {
        return set(new Complex128(value), indices);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CooTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        return CooFieldEquals.cooTensorEquals(this, (CooCTensor) object);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < entries.length; i++) {
            if (!entries[i].isZero()) {
                result = 31*result + entries[i].hashCode();
                result = 31*result + Arrays.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * <p>Formats this sparse COO tensor as a human-readable string specifying the full shape,
     * non-zero entries, and non-zero indices.</p>
     *
     * @return A human-readable string specifying the full shape, non-zero entries, and non-zero indices of this tensor.
     */
    public String toString() {
        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean centring = PrintOptions.useCentering();

        StringBuilder sb = new StringBuilder();

        sb.append("Shape: " + shape + "\n");
        sb.append("Non-zero Entries: " + PrettyPrint.abbreviatedArray(entries, maxCols, padding, precision, centring) + "\n");
        sb.append("Non-zero Indices: " +
                PrettyPrint.abbreviatedArray(indices, PrintOptions.getMaxRows(), maxCols, padding, 20, centring));

        return sb.toString();
    }
}
