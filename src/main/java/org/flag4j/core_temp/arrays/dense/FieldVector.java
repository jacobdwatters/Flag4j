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
import org.flag4j.core_temp.FieldTensorBase;
import org.flag4j.core_temp.VectorMatrixOpsMixin;
import org.flag4j.core_temp.VectorMixin;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.operations.dense.field_ops.DenseFieldEquals;
import org.flag4j.operations.dense.field_ops.DenseFieldTensorDot;
import org.flag4j.operations.dense.field_ops.DenseFieldVectorOperations;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;

// TODO: Needs to implement DenseTensorMixin.
/**
 * <p>A dense vector whose entries are {@link Field field} elements.</p>
 *
 * <p>Vectors are 1D tensors (i.e. rank 1 tensor).</p>
 *
 * <p>FieldVectors have mutable entries but a fixed size.</p>
 *
 * @param <T> Type of the field element for the vector.
 */
public class FieldVector<T extends Field<T>> extends FieldTensorBase<FieldVector<T>, FieldVector<T>, T>
        implements VectorMixin<FieldVector<T>, T>,
        VectorMatrixOpsMixin<FieldVector<T>, FieldMatrix<T>> {

    /**
     * The size of this vector.
     */
    public final int size;


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    protected FieldVector(Shape shape, T[] entries) {
        super(shape, entries);
        ParameterChecks.ensureRank(2, shape);
        size = entries.length;
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
    public FieldVector<T> tensorDot(FieldVector<T> src2, int[] aAxes, int[] bAxes) {
        return DenseFieldTensorDot.tensorDot(this, src2, aAxes, bAxes);
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
    public FieldVector<T> tensorDot(FieldVector<T> src2) {
        return DenseFieldTensorDot.tensorDot(this, src2);
    }


    /**
     * Computes the conjugate transpose of a tensor by conjugating and exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange and conjugate.
     * @param axis2 Second axis to exchange and conjugate.
     *
     * @return The conjugate transpose of this tensor acording to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public FieldVector<T> H(int axis1, int axis2) {
        if(axis1 == axis2 && axis1 == 0) return conj();
        else throw new LinearAlgebraException(String.format("Cannot transpose axes [%d, %d] of tensor with rank 1.", axis1, axis2));
    }


    /**
     * Computes the conjugate transpose of this tensor. That is, conjugates and permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The conjugate transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public FieldVector<T> H(int... axes) {
        return conj();
    }


    /**
     * Creates a vector with the specified entries.
     *
     * @param entries Entries of this vector.
     */
    public FieldVector(T... entries) {
        super(new Shape(entries.length), entries);
        size = entries.length;
    }


    /**
     * Creates a vector with the specified size filled with the {@code fillValue}.
     *
     * @param fillValue Value to fill thes vector with.
     * @param entries Entries of this vector.
     */
    public FieldVector(int size, T fillValue) {
        super(new Shape(size), (T[]) new Field[size]);
        Arrays.fill(this.entries, fillValue);
        this.size = entries.length;
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entires of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public FieldVector<T> makeLikeTensor(Shape shape, T[] entries) {
        return new FieldVector<>(shape, entries);
    }


    /**
     * Joints specified vector with this vector. That is, creates a vector of length {@code this.length() + b.length()} containing
     * first the elements of this vector followed by the elements of {@code b}.
     *
     * @param b Vector to join with this vector.
     *
     * @return A vector resulting from joining the specified vector with this vector.
     */
    @Override
    public FieldVector<T> join(FieldVector<T> b) {
        Field<T>[] joinEntries = new Field[size + b.size];
        System.arraycopy(entries, 0, joinEntries, 0, size);
        System.arraycopy(b.entries, 0, joinEntries, size, b.size);

        return makeLikeTensor(shape, (T[]) joinEntries);
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     *
     * @return The inner product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     */
    @Override
    public T inner(FieldVector<T> b) {
        return DenseFieldVectorOperations.innerProduct(entries, b.entries);
    }


    /**
     * Computes the euclidian norm of this vector.
     *
     * @return The euclidian norm of this vector.
     */
    @Override
    public double norm() {
        return VectorNorms.norm(this);
    }


    /**
     * Computes the p-norm of this vector.
     *
     * @param p {@code p} value in the p-norm.
     *
     * @return The euclidian norm of this vector.
     */
    @Override
    public double norm(int p) {
        return VectorNorms.norm(this, p);
    }


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    @Override
    public FieldVector<T> normalize() {
        double norm = VectorNorms.norm(this);
        return norm==0 ? new FieldVector<T>(size, entries[0].getZero()) : this.div(norm);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     *
     * @return The result of the vector cross product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If either this vector or {@code b} do not have exactly 3 entries.
     */
    @Override
    public FieldVector<T> cross(FieldVector<T> b) {
        ParameterChecks.ensureArrayLengthsEq(3, b.size, this.size);
        Field<T>[] entries = new Field[3];

        entries[0] = this.entries[1].mult(b.entries[2]).sub(this.entries[2].mult(b.entries[1]));
        entries[1] = this.entries[2].mult(b.entries[0]).sub(this.entries[0].mult(b.entries[2]));
        entries[2] = this.entries[0].mult(b.entries[1]).sub(this.entries[1].mult(b.entries[0]));

        return new FieldVector(entries);
    }


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     *
     * @see #isPerp(FieldVector)
     */
    @Override
    public boolean isParallel(FieldVector<T> b) {
        boolean result;

        if(this.size != b.size) {
            result = false;
        } else if(this.size == 1) {
            result = true;
        } else if(this.isZeros() || b.isZeros()) {
            result = true; // Any vector is parallel to zero vector.
        } else {
            result = true;
            Field<T> scale = entries[0].getZero();;

            // Find first non-zero entry of b to compute the scaling factor.
            for(int i = 0; i < b.size; i++) {
                if(!b.entries[i].isZero()) {
                    scale = this.entries[i].div(b.entries[i]);
                    break;
                }
            }

            // Ensure all entries of b are the same scalar multiple of the entries in this vector.
            for(int i = 0; i < this.size; i++) {
                if(!scale.mult(b.entries[i]).equals(this.entries[i])) {
                    result = false;
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Checks if a vector is perpendicular to this vector.
     *
     * @param b Vector to compare to this vector.
     *
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     *
     * @see #isParallel(FieldVector)
     */
    @Override
    public boolean isPerp(FieldVector<T> b) {
        if(this.size != b.size) return false;
        if(this.size == 0) return true;
        return this.inner(b).equals(entries[0].getZero());
    }


    /**
     * Gets the length of a vector.
     *
     * @return The length, i.e. the number of entries, in this vector.
     */
    @Override
    public int length() {
        return size;
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link FieldVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldVector<T> src2 = (FieldVector<T>) object;

        return DenseFieldEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
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
    public FieldVector<T> T(int axis1, int axis2) {
        if(axis1 == axis2 && axis1 == 0) return copy();
        else throw new LinearAlgebraException(String.format("Cannot transpose axes [%d, %d] of tensor with rank 1.", axis1, axis2));
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
    public FieldVector<T> T(int... axes) {
        if(axes.length == 1 && axes[0] == 0) return copy();
        else throw new LinearAlgebraException(String.format("Cannot transpose axes %s of tensor with rank 1.", Arrays.toString(axes)));
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param n Number of times to repeat vector.
     * @param axis Axis along which to repeat vector:
     * <ul>
     *     <li>If {@code axis=0}, then the vector will be treated as a row vector and stacked vertically {@code n} times.</li>
     *     <li>If {@code axis=1} then the vector will be treated as a column vector and stacked horizontaly {@code n} times.</li>
     * </ul>
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    @Override
    public FieldMatrix<T> repeat(int n, int axis) {
        ParameterChecks.ensureValidAxes(shape, axis);
        ParameterChecks.ensureNonNegative(n);

        Field<T>[] tiledEntries = new Field[n*size];
        FieldMatrix<T> tiled;

        if(axis==0) {
            for(int i=0; i<n; i++) // Set each row of the tiled matrix to be the vector values.
                System.arraycopy(entries, 0, tiledEntries, i*size, size);

            tiled = new FieldMatrix<T>(new Shape(n, size), (T[]) tiledEntries);
        } else {
            for(int i=0; i<size; i++) // Fill each row of the tiled matrix with a single value from the vector.
                Arrays.fill(tiledEntries, i*n, (i+1)*n, entries[i]);

            tiled = new FieldMatrix<T>(new Shape(size, n), (T[]) tiledEntries);
        }

        return tiled;
    }


    /**
     * Stacks two vectors vertically as if they were row vectors to form a matrix with two rows.
     *
     * @param b Vector to stack below this vector.
     *
     * @return The result of stacking this vector and vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    @Override
    public FieldMatrix<T> stack(FieldVector<T> b) {
        ParameterChecks.ensureEqualShape(shape, b.shape);

        Field<T>[] tiledEntries = new Field[size];

        // Copy entries from each vector to the matrix.
        System.arraycopy(entries, 0, tiledEntries, 0, size);
        System.arraycopy(b.entries, 0, tiledEntries, size, b.size);

        return new FieldMatrix<T>(2, size, (T[]) tiledEntries);
    }


    /**
     * <p>
     * Stacks two vectors along specified axis.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2-by-n} matrix.
     * </p>
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n-by-2} matrix.
     * </p>
     *
     * @param b VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     * vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     *
     * @return The result of stacking this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public FieldMatrix<T> stack(FieldVector<T> b, int axis) {
        ParameterChecks.ensureAxis2D(axis);
        FieldMatrix<T> stacked;

        if(axis==0) {
            stacked = stack(b);
        } else {
            ParameterChecks.ensureArrayLengthsEq(this.size, b.size);
            Field<T>[] stackedEntries = new Field[2*this.size];

            int count = 0;
            for(int i=0; i<stackedEntries.length; i+=2) {
                stackedEntries[i] = this.entries[count];
                stackedEntries[i+1] = b.entries[count++];
            }

            stacked = new FieldMatrix<T>(this.size, 2, (T[]) stackedEntries);
        }

        return stacked;
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     *
     * @return The result of the vector outer product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public FieldMatrix<T> outer(FieldVector<T> b) {
        return new FieldMatrix<>(shape, (T[]) DenseFieldVectorOperations.outerProduct(entries, b.entries));
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     * <p>If {@code true}, the vector will be converted to a matrix representing a column vector.</p>
     * <p>If {@code false}, The vector will be converted to a matrix representing a row vector.</p>
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public FieldMatrix<T> toMatrix(boolean columVector) {
        if(columVector) {
            return new FieldMatrix<T>(this.entries.length, 1, this.entries.clone()); // Convert to column vector.
        } else {
            return new FieldMatrix<T>(1, this.entries.length, this.entries.clone()); // Convert to row vector.
        }
    }
}
