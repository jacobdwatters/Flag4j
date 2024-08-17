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

package org.flag4j.core_temp.arrays.dense;

import org.flag4j.core.Shape;
import org.flag4j.core_temp.TensorOverField;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.core_temp.structures.fields.RealFloat64;
import org.flag4j.core_temp.structures.fields.utils.CompareField;
import org.flag4j.operations.dense.field_ops.DenseFieldEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;


/**
 * A tensor whose entries are field elements.
 * @param <T> Type of the field element for the tensor.
 */
public class FieldTensor<T extends Field<T>> extends TensorOverField<FieldTensor<T>, T[], T> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public FieldTensor(Shape shape, T[] entries) {
        super(shape, entries);
    }


    /**
     * Gets the element of this tensor at the specified indices.
     *
     * @param indices Indices of the element to get.
     *
     * @return The element of this tensor at the specified indices.
     *
     * @throws IllegalArgumentException       If {@code indices} is not of length 2.
     * @throws ArrayIndexOutOfBoundsException If any indices are not within this matrix.
     */
    @Override
    public T get(int... indices) {
        ParameterChecks.assertValidIndex(shape, indices);
        return entries[shape.entriesIndex(indices)];
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public FieldTensor<T> flatten() {
        return new FieldTensor(new Shape(entries.length), entries.clone());
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
    public FieldTensor<T> flatten(int axis) {
        int[] dims = new int[this.getRank()];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntries().intValueExact();
        Shape flatShape = new Shape(true, dims);

        return new FieldTensor(flatShape, entries.clone());
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
    public FieldTensor<T> add(FieldTensor<T> b) {
        ParameterChecks.assertEqualShape(shape, b.shape);
        Field<T>[] sum = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            sum[i] = entries[i].add(b.entries[i]);
        }

        return new FieldTensor(shape, sum);
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
    @Override
    public FieldTensor<T> sub(FieldTensor<T> b) {
        ParameterChecks.assertEqualShape(shape, b.shape);
        Field<T>[] diff = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            diff[i] = entries[i].sub(b.entries[i]);
        }

        return new FieldTensor(shape, diff);
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public FieldTensor<RealFloat64> abs() {
        Field<RealFloat64>[] conj = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            conj[i] = new RealFloat64(entries[i].abs());
        }

        return new FieldTensor(shape, conj);
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
    public FieldTensor<T> elemMult(FieldTensor<T> b) {
        ParameterChecks.assertEqualShape(shape, b.shape);
        Field<T>[] diff = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            diff[i] = entries[i].sub(b.entries[i]);
        }

        return new FieldTensor(shape, diff);
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
    public FieldTensor<T> tensorDot(FieldTensor<T> src2, int[] aAxes, int[] bAxes) {
        return null;
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
    public FieldTensor<T> tensorDot(FieldTensor<T> src2) {
        return null;
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specifieed by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.</p>
     *
     * @param axis1
     * @param axis2
     *
     * @return The generalized trace of this tensor along @link axis1} and {@code axis2}.
     */
    @Override
    public FieldTensor<T> tensorTr(int axis1, int axis2) {
        return null;
    }


    /**
     * Adds a sclar field value to each entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public FieldTensor<T> add(T b) {
        Field<T>[] sum = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            sum[i] = entries[i].add(b);
        }

        return new FieldTensor(shape, sum);
    }


    /**
     * Subtracts a sclar field value from each entry of this tensor.
     *
     * @param b Scalar field value in differencce.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public FieldTensor<T> sub(T b) {
        Field<T>[] diff = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            diff[i] = entries[i].sub(b);
        }

        return new FieldTensor(shape, diff);
    }


    /**
     * Multiplies a sclar field value to each entry of this tensor.
     *
     * @param b Scalar field value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    public FieldTensor<T> mult(T b) {
        Field<T>[] product = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            product[i] = entries[i].mult(b);
        }

        return new FieldTensor(shape, product);
    }


    /**
     * Divides each entry of this tensor by s scalar field element.
     *
     * @param b Scalar field value in quotient.
     *
     * @return The quotient of this tensor with {@code b}.
     */
    @Override
    public FieldTensor<T> div(T b) {
        Field<T>[] product = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            product[i] = entries[i].div(b);
        }

        return new FieldTensor(shape, product);
    }


    /**
     * Finds the minimum value in this tensor.
     *
     * @return The minimum value in this tensor.
     */
    @Override
    public T min() {
        return CompareField.min(entries);
    }


    /**
     * Finds the maximum value in this tensor.
     *
     * @return The maximum value in this tensor.
     */
    @Override
    public T max() {
        return CompareField.max(entries);
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public double minAbs() {
        return CompareField.minAbs(entries);
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public double maxAbs() {
        return CompareField.maxAbs(entries);
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        return new int[0];
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        return new int[0];
    }


    /**
     * Finds the indices of the minimum absollte value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMinAbs() {
        return new int[0];
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMaxAbs() {
        return new int[0];
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public FieldTensor<T> sqrt() {
        Field<T>[] sqrt = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            sqrt[i] = entries[i].sqrt();
        }

        return new FieldTensor(shape, sqrt);
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
    public FieldTensor<T> T() {
        return T(0, getRank()-1);
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor acording to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public FieldTensor<T> T(int axis1, int axis2) {
        // TODO: Need dispatcher.
        if(getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new LinearAlgebraException("TensorOld transpose not defined for rank " + getRank() + " tensor.");
        }

        Field<T>[] dest = new Field[shape.totalEntries().intValue()];
        Shape destShape = shape.swapAxes(axis1, axis2);
        int[] destIndices;

        for(int i=0, size=entries.length; i<size; i++) {
            destIndices = shape.getIndices(i);
            ArrayUtils.swap(destIndices, axis1, axis2); // Compute destination indices.
            dest[destShape.entriesIndex(destIndices)] = entries[i]; // Apply transpose for the element.
        }

        return new FieldTensor(destShape, dest);
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
    public FieldTensor<T> T(int... axes) {
        // TODO: Need dispatcher.
        ParameterChecks.assertPermutation(axes);
        ParameterChecks.assertEquals(shape.getRank(), axes.length);
        if(shape.getRank() < 2) { // Can't transpose tensor with less than 2 axes.
            throw new LinearAlgebraException("TensorOld transpose not defined for rank " + shape.getRank() + " tensor.");
        }

        Field<T>[] dest = new Field[shape.totalEntries().intValue()];
        Shape destShape = shape.swapAxes(axes);
        int[] destIndices;

        for(int i=0, size=entries.length; i<entries.length; i++) {
            destIndices = shape.getIndices(i);
            ArrayUtils.swap(destIndices, axes); // Compute destination indices.
            dest[destShape.entriesIndex(destIndices)] = entries[i]; // Apply transpose for the element.
        }

        return new FieldTensor(destShape, dest);
    }


    /**
     * Computes the element-wise reciprocals of this tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     */
    @Override
    public FieldTensor<T> recip() {
        Field<T>[] recip = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            recip[i] = entries[i].multInv();
        }

        return new FieldTensor(shape, recip);
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public FieldTensor<T> copy() {
        return new FieldTensor(shape, entries.clone());
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link Tensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldTensor<T> src2 = (FieldTensor<T>) object;

        return DenseFieldEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
    }
}
