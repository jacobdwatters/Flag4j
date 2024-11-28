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
import org.flag4j.arrays.backend.field.AbstractCooFieldTensor;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.common.complex.Complex128Ops;
import org.flag4j.linalg.operations.common.complex.Complex128Properties;
import org.flag4j.linalg.operations.dense.real.RealDenseTranspose;
import org.flag4j.linalg.operations.sparse.coo.field_ops.CooFieldEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>Sparse complex tensor stored in coordinate list (COO) format. The data of this COO tensor are of type {@link Complex128}</p>
 *
 * <p>The non-zero data and non-zero indices of a COO tensor are mutable but the {@link #shape} and total number of
 * {@link #data non-zero data} is fixed.</p>
 *
 * <p>Sparse tensors allow for the efficient storage of and operations on tensors that contain many zero values.</p>
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e. tensors which contain almost all zeros relative to the size of the
 * tensor).</p>
 *
 * <p>A sparse COO tensor is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #data} of the tensor. All other data in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #data}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many operations assume indices to be sorted in a
 *     row-major format (i.e. last index increased fastest) but often this is not explicitly verified.</p>
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero data in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the ND
 *     index of {@code data[i]}.</p>
 *     </li>
 * </ul>
 */
public class CooCTensor extends AbstractCooFieldTensor<CooCTensor, CTensor, Complex128> {

    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
     */
    public CooCTensor(Shape shape, Field<Complex128>[] entries, int[][] indices) {
        super(shape, entries, indices);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
     */
    public CooCTensor(Shape shape, List<Field<Complex128>> entries, List<int[]> indices) {
        super(shape, entries.toArray(new Complex128[0]), indices.toArray(new int[0][]));
        if(super.data.length == 0 || super.data[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
     */
    public CooCTensor(Shape shape, Complex64[] entries, int[][] indices) {
        super(shape, new Complex128[entries.length], indices);
        setZeroElement(Complex128.ZERO);

        for(int i=0, size=entries.length; i<size; i++)
            this.data[i] = new Complex128((Complex64) entries[i]);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
     */
    public CooCTensor(Shape shape) {
        super(shape, new Complex128[0], new int[0][shape.getRank()]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor.
     * @param indices Indices of the non-zero data of this tensor.
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
        super(b.shape, b.data.clone(), ArrayUtils.deepCopy(b.indices, null));
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
    public CooCTensor makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CooCTensor(shape, entries, ArrayUtils.deepCopy(indices, null));
    }


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero data.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Non-zero data of the tensor to construct.
     * @param indices Indices of the non-zero data of the tensor.
     *
     * @return A tensor of the same type as this tensor with the specified shape and non-zero data.
     */
    @Override
    public CooCTensor makeLikeTensor(Shape shape, Complex128[] entries, int[][] indices) {
        return new CooCTensor(shape, entries, indices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero data.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Non-zero data of the tensor to construct.
     * @param indices Indices of the non-zero data of the tensor.
     *
     * @return A tensor of the same type as this tensor with the specified shape and non-zero data.
     */
    @Override
    public CooCTensor makeLikeTensor(Shape shape, List<Field<Complex128>> entries, List<int[]> indices) {
        return new CooCTensor(shape, entries, indices);
    }


    /**
     * Constructs a dense tensor that is a similar type as this sparse COO tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries The data of the dense tensor to construct.
     *
     * @return A dense tensor that is a similar type as this sparse COO tensor.
     */
    @Override
    public CTensor makeLikeDenseTensor(Shape shape, Field<Complex128>[] entries) {
        return new CTensor(shape, entries);
    }


    /**
     * Converts this complex vector to a real vector.
     * @return A real vector containing the real components of all non-zero values in this vector. The imaginary components are
     * ignored.
     */
    public CooTensor toReal() {
        return new CooTensor(shape, Complex128Ops.toReal(data), indices.clone());
    }


    /**
     * Checks if all data of this matrix are real.
     * @return {@code true} if all data of this matrix are real. Otherwise, returns {@code false}.
     */
    public boolean isReal() {
        return Complex128Properties.isReal(data);
    }


    /**
     * Checks if any entry within this matrix has non-zero imaginary component.
     * @return {@code true} if any entry of this matrix has a non-zero imaginary component.
     */
    public boolean isComplex() {
        return Complex128Properties.isComplex(data);
    }


    /**
     * Rounds all data within this tensor to the specified precision.
     * @param precision The precision to round to (i.e. the number of decimal places to round to). Must be non-negative.
     * @return A new tensor containing the data of this tensor rounded to the specified precision.
     */
    public CooCTensor round(int precision) {
        return new CooCTensor(shape, Complex128Ops.round(data, precision), ArrayUtils.deepCopy(indices, null));
    }


    /**
     * Sets all elements of this tensor to zero if they are within {@code tol} of zero. This is <i>not</i> done in place.
     * @param precision The precision to round to (i.e. the number of decimal places to round to). Must be non-negative.
     * @return A copy of this tensor with all data within {@code tol} of zero set to zero.
     */
    public CooCTensor roundToZero(double tolerance) {
        Complex128[] rounded = Complex128Ops.roundToZero(data, tolerance);
        List<Field<Complex128>> dest = new ArrayList<>(data.length);
        List<int[]> destIndices = new ArrayList<>(data.length);

        for(int i = 0, size = data.length; i<size; i++) {
            if(!rounded[i].isZero()) {
                dest.add(rounded[i]);
                destIndices.add(indices[i].clone());
            }
        }

        return new CooCTensor(shape, dest, destIndices);
    }


    /**
     * Converts this tensor to a matrix with specified shape.
     * @param newShape Shape of matrix to convert this tensor to. Shape must be broadcastable with this tensors shape and have rank 2.
     * @return A matrix of the specified shape with the same non-zero data as this tensor.
     */
    public CooCMatrix toMatrix(Shape newShape) {
        ValidateParameters.ensureRank(newShape, 2);
        CooCTensor t = reshape(newShape); // Reshape as rank 2 tensor. Broadcastable check made here.
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(t.indices);

        return new CooCMatrix(newShape, t.data.clone(), tIndices[0], tIndices[1]);
    }


    /**
     * Sets the element of this tensor at the specified target index.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param target Index of the element to set.
     *
     * @return If this tensor is dense, a reference to this tensor is returned. If this tensor is sparse, a copy of this tensor with
     * the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code target} is not within the bounds of this tensor.
     */
    public CooCTensor set(double value, int... target) {
        return super.set(new Complex128(value), target);
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
            if (!data[i].isZero()) {
                result = 31*result + data[i].hashCode();
                result = 31*result + Arrays.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * <p>Formats this sparse COO tensor as a human-readable string specifying the full shape,
     * non-zero data, and non-zero indices.</p>
     *
     * @return A human-readable string specifying the full shape, non-zero data, and non-zero indices of this tensor.
     */
    public String toString() {
        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean centring = PrintOptions.useCentering();

        StringBuilder sb = new StringBuilder();

        sb.append("Shape: " + shape + "\n");
        sb.append("Non-zero Entries: " + PrettyPrint.abbreviatedArray(data, maxCols, padding, precision, centring) + "\n");
        sb.append("Non-zero Indices: " +
                PrettyPrint.abbreviatedArray(indices, PrintOptions.getMaxRows(), maxCols, padding, 20, centring));

        return sb.toString();
    }
}
