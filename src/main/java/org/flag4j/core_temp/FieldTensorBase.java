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

package org.flag4j.core_temp;

import org.flag4j.core.Shape;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.operations.common.field_ops.AggregateField;
import org.flag4j.operations.common.field_ops.CompareField;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;

/**
 * The base class for all tensors whose entries are elements of a {@link Field}.
 *
 * @param <T> The type of this tensor.
 * @param <U> Type of dense tensor equivalent to {@code T}. If {@code T} is dense, then this should be the same type as {@code T}.
 * This parameter is required because some operations (e.g. {@link #tensorDot(TensorOverSemiRing, int)}) between two sparse tensors
 * result in a dense tensor.
 * @param <V> The type of the {@link Field} for this tensor's entries.
 */
public abstract class FieldTensorBase<T extends FieldTensorBase<T, U, V>,
        U extends FieldTensorBase<U, U, V>, V extends Field<V>> extends TensorOverField<T, U, V[], V> {

    /**
     * Stores the zero element of the field for this tensor.
     */
    private V zeroElement = null;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected FieldTensorBase(Shape shape, V[] entries) {
        super(shape, entries);
        if(entries.length > 0 && entries[0] != null) zeroElement = entries[0].getZero();
    }


    /**
     * Sets the zero element for the field of this tensor. This is useful in some operations for cases where the total number of
     * entries or total number of non-zero entries is zero. In such cases, the zero element cannot be determined for a generic field so
     * {@code null} is used. When
     * @param zeroElement Zero element for the field of this tensor.
     */
    protected void setZeroElement(V zeroElement) {
        if(!this.zeroElement.isZero())
            throw new IllegalArgumentException("zeroElement is not an additive identity for the Field of this tensor.");

        this.zeroElement = zeroElement;
    }


    /**
     * Gets the zero element for the field of this tensor.
     * @return The zero element for the field of this tensor. If it could not be determined during construction of this object
     * and has not been set explicitly by {@link #setZeroElement(Field)} then {@code null} will be returned.
     */
    protected V getZeroElement() {
        return zeroElement;
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
    public V get(int... indices) {
        ParameterChecks.ensureValidIndex(shape, indices);
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
    public T flatten() {
        return makeLikeTensor(new Shape(entries.length), entries.clone());
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
    public T flatten(int axis) {
        int[] dims = new int[this.getRank()];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntries().intValueExact();
        Shape flatShape = new Shape(dims);

        return makeLikeTensor(flatShape, entries.clone());
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
        ParameterChecks.ensureBroadcastable(this.shape, shape);
        return makeLikeTensor(shape, this.entries.clone());
    }


    /**
     * Computes the sum of all values in this tensor.
     *
     * @return The sum of all values in this tensor.
     */
    @Override
    public V sum() {
        if(entries.length==0) return zeroElement;
        return AggregateField.sum(entries);
    }


    /**
     * Computes the product of all values in this tensor.
     *
     * @return The product of all values in this tensor.
     */
    @Override
    public V prod() {
        if(entries.length==0) return zeroElement;
        return AggregateField.prod(entries);
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
    public T add(T b) {
        ParameterChecks.ensureEqualShape(shape, b.shape);
        Field<V>[] sum = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            sum[i] = entries[i].add(b.entries[i]);

        return makeLikeTensor(shape, (V[]) sum);
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
    public T sub(T b) {
        ParameterChecks.ensureEqualShape(shape, b.shape);
        Field<V>[] diff = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            diff[i] = entries[i].sub(b.entries[i]);

        return makeLikeTensor(shape, (V[]) diff);
    }


    /**
     * Adds a real value to each entry of this tensor.
     *
     * @param b Value to add to each value of this tensor.
     *
     * @return Sum of this tensor with {@code b}.
     */
    @Override
    public T add(double b) {
        Field<V>[] sum = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            sum[i] = entries[i].add(b);

        return makeLikeTensor(shape, (V[]) sum);
    }


    /**
     * Subtracts a real value from each entry of this tensor.
     *
     * @param b Value to subtract from each value of this tensor.
     *
     * @return Difference of this tensor with {@code b}.
     */
    @Override
    public T sub(double b) {
        Field<V>[] diff = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            diff[i] = entries[i].sub(b);

        return makeLikeTensor(shape, (V[]) diff);
    }


    /**
     * Multiplies a real value to each entry of this tensor.
     *
     * @param b Value to multiply to each value of this tensor.
     *
     * @return Product of this tensor with {@code b}.
     */
    @Override
    public T mult(double b) {
        Field<V>[] product = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            product[i] = entries[i].mult(b);

        return makeLikeTensor(shape, (V[]) product);
    }


    /**
     * Divides each entry of this tensor by a real value.
     *
     * @param b Value to divide each value of this tensor by.
     *
     * @return Quotient of this tensor with {@code b}.
     */
    @Override
    public T div(double b) {
        Field<V>[] quotient = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            quotient[i] = entries[i].div(b);

        return makeLikeTensor(shape, (V[]) quotient);
    }


    /**
     * Computes the element-wise conjugation of this tensor.
     *
     * @return The element-wise conjugation of this tensor.
     */
    @Override
    public T conj() {
        Field<V>[] conj = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            conj[i] = entries[i].conj();

        return makeLikeTensor(shape, (V[]) conj);
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
        ParameterChecks.ensureEqualShape(shape, b.shape);
        Field<V>[] prod = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            prod[i] = entries[i].mult(b.entries[i]);

        return makeLikeTensor(shape, (V[]) prod);
    }


    /**
     * <p>Computes the generalized trace of this tensor along the specified axes.</p>
     *
     * <p>The generalized tensor trace is the sum along the diagonal values of the 2D sub-arrays_old of this tensor specified by
     * {@code axis1} and {@code axis2}. The shape of the resulting tensor is equal to this tensor with the
     * {@code axis1} and {@code axis2} removed.</p>
     *
     * @param axis1 First axis for 2D sub-array.
     * @param axis2 Second axis for 2D sub-array.
     *
     * @return The generalized trace of this tensor along {@code axis1} and {@code axis2}.
     * @throws IndexOutOfBoundsException If the two axes are not both larger than zero and less than this tensors rank.
     * @throws IllegalArgumentException If {@code axis1 == @code axis2} or {@code this.shape.get(axis1) != this.shape.get(axis1)}
     * (i.e. the axes are equal or the tensor does not have the same length along the two axes.)
     */
    @Override
    public T tensorTr(int axis1, int axis2) {
        ParameterChecks.ensureNotEquals(axis1, axis2);
        ParameterChecks.ensureValidIndices(getRank(), axis1, axis2);
        ParameterChecks.ensureEquals(shape.get(axis1), shape.get(axis2));

        int[] strides = shape.getStrides();
        int rank = strides.length;
        int[] newDims = new int[rank - 2];

        int idx = 0;

        // Compute shape for resulting tensor.
        for(int i=0; i<rank; i++) {
            if(i != axis1 && i != axis2) newDims[idx++] = shape.get(i);
        }

        Shape destShape = new Shape(newDims);
        Field<V>[] destEntries = new Field[destShape.totalEntries().intValueExact()];

        // Calculate the offset increment for the diagonal.
        int traceLength = shape.get(axis1);
        int diagonalStride = strides[axis1] + strides[axis2];

        int[] destIndices = new int[rank - 2];
        for(int i=0; i<destEntries.length; i++) {
            destIndices = destShape.getIndices(i);

            int baseOffset = 0;
            idx = 0;

            // Compute offset for mapping destination indices to indices in this tensor.
            for(int j=0; j<rank; j++) {
                if(j != axis1 && j != axis2) {
                    baseOffset += destIndices[idx++]*strides[j];
                }
            }

            // Sum over diagonal elements of the 2D sub-array.
            V sum = zeroElement;
            int offset = baseOffset;
            for(int diag=0; diag<traceLength; diag++) {
                sum = sum.add(entries[offset]);
                offset += diagonalStride;
            }

            destEntries[i] = sum;
        }

        return makeLikeTensor(destShape, (V[]) destEntries);
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZeros() {
        return entries.length==0 || CompareField.isZeros(entries);
    }


    /**
     * Adds a scalar field value to each entry of this tensor.
     *
     * @param b Scalar field value in sum.
     *
     * @return The sum of this tensor with the scalar {@code b}.
     */
    @Override
    public T add(V b) {
        Field<V>[] sum = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            sum[i] = entries[i].add(b);

        return makeLikeTensor(shape, (V[]) sum);
    }


    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar field value in sum.
     */
    @Override
    public void addEq(V b) {
        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].add(b);
    }


    /**
     * Subtracts a scalar field value from each entry of this tensor.
     *
     * @param b Scalar field value in difference.
     *
     * @return The difference of this tensor and the scalar {@code b}.
     */
    @Override
    public T sub(V b) {
        Field<V>[] diff = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            diff[i] = entries[i].sub(b);

        return makeLikeTensor(shape, (V[]) diff);
    }


    /**
     * Subtracts a scalar value from each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in difference.
     */
    @Override
    public void subEq(V b) {
        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].sub(b);
    }


    /**
     * Multiplies a scalar field value to each entry of this tensor.
     *
     * @param b Scalar field value in product.
     *
     * @return The product of this tensor with {@code b}.
     */
    @Override
    public T mult(V b) {
        Field<V>[] product = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            product[i] = entries[i].mult(b);

        return makeLikeTensor(shape, (V[]) product);
    }


    /**
     * Multiplies a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in product.
     */
    @Override
    public void multEq(V b) {
        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].mult(b);
    }


    /**
     * Divides each entry of this tensor by a scalar field element.
     *
     * @param b Scalar field value in quotient.
     *
     * @return The quotient of this tensor with {@code b}.
     */
    @Override
    public T div(V b) {
        Field<V>[] quotient = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            quotient[i] = entries[i].div(b);

        return makeLikeTensor(shape, (V[]) quotient);
    }


    /**
     * Divides each entry of this tensor by s scalar field element and stores the result in this tensor.
     *
     * @param b Scalar field value in quotient.
     */
    @Override
    public void divEq(V b) {
        for(int i=0, size=entries.length; i<size; i++)
            entries[i] = entries[i].div(b);
    }


    /**
     * Finds the minimum value in this tensor.
     *
     * @return The minimum value in this tensor.
     */
    @Override
    public V min() {
        return CompareField.min(entries);
    }


    /**
     * Finds the maximum value in this tensor.
     *
     * @return The maximum value in this tensor.
     */
    @Override
    public V max() {
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
    public int[] argmin() {
        return shape.getIndices(CompareField.argmin());
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmax() {
        return shape.getIndices(CompareField.argmax());
    }


    /**
     * Finds the indices of the minimum absolute value in this tensor.
     *
     * @return The indices of the minimum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argminAbs() {
        return shape.getIndices(CompareField.argminAbs());
    }


    /**
     * Finds the indices of the maximum absolute value in this tensor.
     *
     * @return The indices of the maximum absolute value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argmaxAbs() {
        return shape.getIndices(CompareField.argmaxAbs());
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public T sqrt() {
        Field<V>[] sqrt = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            sqrt[i] = entries[i].sqrt();

        return makeLikeTensor(shape, (V[]) sqrt);
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
        return T(0, getRank()-1);
    }


    /**
     * Computes the Hermitian transpose of a tensor by exchanging and conjugating the first and last axes of this tensor.
     *
     * @return The Hermitian transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public T H() {
        return H(0, getRank()-1);
    }


    /**
     * Computes the element-wise reciprocals of this tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     */
    @Override
    public T recip() {
        Field<V>[] recip = new Field[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            recip[i] = entries[i].multInv();

        return makeLikeTensor(shape, (V[]) recip);
    }


    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public T copy() {
        return makeLikeTensor(shape, entries.clone());
    }
}
