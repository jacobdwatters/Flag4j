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
import org.flag4j.arrays.backend_new.field.AbstractCooFieldTensor;
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
public class CooCTensor extends AbstractCooFieldTensor<CooCTensor, CTensor, Complex128> {

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
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Non-zero entries of the tensor to construct.
     * @param indices Indices of the non-zero entries of the tensor.
     *
     * @return A tensor of the same type as this tensor with the specified shape and non-zero entries.
     */
    @Override
    public CooCTensor makeLikeTensor(Shape shape, Complex128[] entries, int[][] indices) {
        return new CooCTensor(shape, entries, indices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Non-zero entries of the tensor to construct.
     * @param indices Indices of the non-zero entries of the tensor.
     *
     * @return A tensor of the same type as this tensor with the specified shape and non-zero entries.
     */
    @Override
    public CooCTensor makeLikeTensor(Shape shape, List<Field<Complex128>> entries, List<int[]> indices) {
        return new CooCTensor(shape, entries, indices);
    }


    /**
     * Constructs a dense tensor that is a similar type as this sparse COO tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries The entries of the dense tensor to construct.
     *
     * @return A dense tensor that is a similar type as this sparse COO tensor.
     */
    @Override
    public CTensor makeLikeDenseTensor(Shape shape, Field<Complex128>[] entries) {
        return new CTensor(shape, entries);
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
    public CooCTensor makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CooCTensor(shape, entries, ArrayUtils.deepCopy(indices, null));
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
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CooCTensor}.
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

        for(int i=0; i<nnz; i++) {
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
