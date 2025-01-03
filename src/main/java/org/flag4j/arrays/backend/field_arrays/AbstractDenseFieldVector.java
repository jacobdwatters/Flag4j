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

package org.flag4j.arrays.backend.field_arrays;

import org.flag4j.algebraic_structures.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.dense.DenseConcat;
import org.flag4j.linalg.ops.dense.field_ops.DenseFieldVectorOps;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringVectorOps;
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
        extends AbstractDenseFieldTensor<T, V>
        implements VectorMixin<T, U, U, V> {

    /**
     * The size of this vector. This is the total number of data stored in this vector.
     */
    public final int size;


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractDenseFieldVector(Shape shape, V[] entries) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 1);
        size = entries.length;
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
     * Constructs a dense vector with the specified {@code data} of the same type as the vector.
     * @param entries Entries of the dense vector to construct.
     */
    public abstract T makeLikeTensor(V[] entries);


    /**
     * Constructs a matrix of similar type to this vector with the specified {@code shape} and {@code data}.
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     * @return A matrix of similar type to this vector with the specified {@code shape} and {@code data}.
     */
    public abstract U makeLikeMatrix(Shape shape, V[] entries);


    /**
     * Joints specified vector with this vector. That is, creates a vector of length {@code this.length() + b.length()} containing
     * first the elements of this vector followed by the elements of {@code b}.
     *
     * @param b Vector to join with this vector.
     *
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public T join(T b) {
        V[] dest = makeEmptyDataArray(size + b.size);
        DenseConcat.concat(data, b.data, dest);
        return makeLikeTensor(dest);
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
     * <p>Computes the dot product between two vectors.
     *
     * <p>Note: this method is distinct from {@link #inner(AbstractDenseFieldVector)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     * @see #inner(AbstractDenseFieldVector)
     */
    @Override
    public V dot(T b) {
        return DenseSemiringVectorOps.dotProduct(data, b.data);
    }


    /**
     * Gets the length of a vector. Same as {@link #size()}.
     *
     * @return The length, i.e. the number of data, in this vector.
     */
    @Override
    public int length() {
        return size;
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param n Number of times to repeat vector. Must be positive.
     * @param axis Axis along which to repeat vector. Must be either 1 or 0.
     * <ul>
     *     <li>If {@code axis=0}, then the vector will be treated as a row vector and stacked vertically {@code n} times.</li>
     *     <li>If {@code axis=1} then the vector will be treated as a column vector and stacked horizontally {@code n} times.</li>
     * </ul>
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    @Override
    public U repeat(int n, int axis) {
        V[] dest = makeEmptyDataArray(size*n);
        DenseConcat.repeat(data, n, axis, dest); // axis is verified to be 1 or 0 here.
        Shape shape = (axis==0) ? new Shape(n, size) : new Shape(size, n);

        return makeLikeMatrix(shape, dest);
    }


    /**
     * <p>Stacks two vectors along specified axis.
     *
     * <p>Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2-by-n} matrix.
     *
     * <p>Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n-by-2} matrix.
     *
     * @param b Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of data in this vector is different from the number of
     *                                  data in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public U stack(T b, int axis) {
        V[] dest = makeEmptyDataArray(2*size);
        DenseConcat.stack(data, b.data, axis, dest);
        Shape shape = (axis==0) ? new Shape(2, size) : new Shape(size, 2);
        return makeLikeMatrix(shape, dest);
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
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     * <ul>
     *     <li>If {@code true}, the vector will be converted to a matrix representing a column vector.</li>
     *     <li>If {@code false}, The vector will be converted to a matrix representing a row vector.</li>
     * </ul>
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public U toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector.
            return makeLikeMatrix(new Shape(data.length, 1), data.clone());
        } else {
            // Convert to row vector.
            return makeLikeMatrix(new Shape(1, data.length), data.clone());
        }
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
        return div(mag());
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
     * Gets the element of this vector at the specified index.
     *
     * @param idx Index of the element to get within this vector.
     *
     * @return The element of this vector at index {@code idx}.
     */
    @Override
    public V get(int idx) {
        ValidateParameters.validateTensorIndex(shape, idx);
        return data[idx];
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
    public double norm(int p) {
        return VectorNorms.norm(data, p);
    }
}
