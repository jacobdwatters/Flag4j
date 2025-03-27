/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.arrays.backend.field_arrays;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.backend.ring_arrays.AbstractDenseRingVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.common.ring_ops.RingProperties;
import org.flag4j.linalg.ops.dense.field_ops.DenseFieldElemDiv;
import org.flag4j.linalg.ops.dense.field_ops.DenseFieldVectorOps;
import org.flag4j.numbers.Field;
import org.flag4j.util.ValidateParameters;


/**
 * <p>The base class for all dense vectors whose data are {@link Field} elements.
 *
 * <p>Vectors are 1D tensors (i.e. rank 1 tensor).
 *
 * <p>AbstractDenseFieldVectors have mutable {@link #data} but a fixed {@link #shape}.
 *
 * @param <T> Type of the vector.
 * @param <U> Type of matrix equivalent to this vector.
 * @param <V> Type of the {@link Field field} element of this vector.
 */
public abstract class AbstractDenseFieldVector<T extends AbstractDenseFieldVector<T, U, V>,
        U extends AbstractDenseFieldMatrix<U, T, V>, V extends Field<V>>
        extends AbstractDenseRingVector<T, U, V>
        implements VectorMixin<T, U, U, V>, FieldTensorMixin<T, T, V> {


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractDenseFieldVector(Shape shape, V[] entries) {
        super(shape, entries);
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor according to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public T H(int axis1, int axis2) {
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);
        return conj();
    }


    /**
     * <p>Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     * @see #dot(AbstractDenseFieldVector)
     */
    @Override
    public V inner(T b) {
        return DenseFieldVectorOps.innerProduct(data, b.data);
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of data.
     */
    @Override
    public U outer(T b) {
        V[] dest = makeEmptyDataArray(size*b.size);
        DenseFieldVectorOps.outerProduct(data, b.data, dest);
        return makeLikeMatrix(new Shape(size, b.size), dest);
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public Vector abs() {
        double[] abs = new double[data.length];
        RingOps.abs(data, abs);
        return new Vector(shape, abs);
    }


    /**
     * Normalizes this vector to a unit length vector.
     *
     * @return This vector normalized to a unit length.
     */
    @Override
    public T normalize() {
        V[] dest = makeEmptyDataArray(size);
        FieldOps.div(data, mag(), dest);
        return makeLikeTensor(shape, dest);
    }


    /**
     * Computes the magnitude of this vector.
     *
     * @return The magnitude of this vector.
     */
    @Override
    public V mag() {
        V mag = getZeroElement();

        for(int i=0; i<size; i++)
            mag = mag.add(data[i].mult( data[i]));

        return mag.sqrt();
    }


    /**
     * Computes the Euclidean norm of this vector.
     *
     * @return The Euclidean norm of this vector.
     */
    public double norm() {
        return VectorNorms.norm(data);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The Euclidean norm of this vector.
     */
    public double norm(double p) {
        return VectorNorms.norm(data, p);
    }


    /**
     * Computes the element-wise quotient of two matrices.
     *
     * @param b Second matrix in the element-wise quotient.
     *
     * @return The element-wise quotient of this matrix and {@code b}.
     */
    @Override
    public T div(T b) {
        V[] dest = makeEmptyDataArray(data.length);
        DenseFieldElemDiv.dispatch(data, shape, b.data, b.shape, dest);
        return makeLikeTensor(shape, dest);
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        V[] dest = makeEmptyDataArray(data.length);
        FieldOps.sqrt(data, dest);
        return makeLikeTensor(shape, dest);
    }


    /**
     * Checks if this tensor only contains finite values.
     *
     * @return {@code true} if this tensor only contains finite values; {@code false} otherwise.
     *
     * @see #isInfinite()
     * @see #isNaN()
     */
    @Override
    public boolean isFinite() {
        return FieldOps.isFinite(data);
    }


    /**
     * Checks if this tensor contains at least one infinite value.
     *
     * @return {@code true} if this tensor contains at least one infinite value; {@code false} otherwise.
     *
     * @see #isFinite()
     * @see #isNaN()
     */
    @Override
    public boolean isInfinite() {
        return FieldOps.isInfinite(data);
    }


    /**
     * Checks if this tensor contains at least one NaN value.
     *
     * @return {@code true} if this tensor contains at least one NaN value; {@code false} otherwise.
     *
     * @see #isFinite()
     * @see #isInfinite()
     */
    @Override
    public boolean isNaN() {
        return FieldOps.isInfinite(data);
    }


    /**
     * Checks if all data of this matrix are 'close' as defined below. Custom tolerances may be specified using
     * {@link #allClose(AbstractDenseFieldVector, double, double)}.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same shape and all data are 'close' element-wise, i.e.
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (1E-08 + 1E-05*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDenseFieldVector, double, double) 
     */
    public boolean allClose(T b) {
        return sameShape(b) && RingProperties.allClose(data, b.data);
    }


    /**
     * Checks if all data of this matrix are 'close' as defined below.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same length and all data are 'close' element-wise, i.e.
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (absTol + relTol*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDenseFieldVector)
     */
    public boolean allClose(T b, double relTol, double absTol) {
        return sameShape(b) && RingProperties.allClose(data, b.data, relTol, absTol);
    }
}
