/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

package com.flag4j.core;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies operations which should be implemented by any vector.
 * @param <T> Vector type.
 * @param <U> Dense Vector type.
 * @param <V> Sparse Vector type.
 * @param <W> Complex Vector type.
 * @param <X> Vector entry type.
 * @param <TT> Matrix type equivalent.
 * @param <UU> Dense Matrix type equivalent.
 * @param <WW> Complex Matrix type equivalent.
 */
public interface VectorOperationsMixin<T, U, V, W, X extends Number, TT, UU, WW> {


    /**
     * Joints specified vector with this vector.
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    U join(Vector b);


    /**
     * Joints specified vector with this vector.
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    CVector join(CVector b);


    /**
     * Joints specified vector with this vector.
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    T join(SparseVector b);


    /**
     * Joints specified vector with this vector.
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    W join(SparseCVector b);

    // TODO: Add stack(vec, axis) methods so vectors can be stacked as if column vectors.

    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     * - If the number of entries in this vector is different from the number of entries in
     * the vector {@code b}.
     */
    TT stack(Vector b);


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     * - If the number of entries in this vector is different from the number of entries in
     * the vector {@code b}.
     */
    TT stack(SparseVector b);


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     * - If the number of entries in this vector is different from the number of entries in
     * the vector {@code b}.
     */
    WW stack(CVector b);


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     * - If the number of entries in this vector is different from the number of entries in
     * the vector {@code b}.
     */
    WW stack(SparseCVector b);


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
     * @param b Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     * entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    TT stack(Vector b, int axis);


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
     * @param b Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     * entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    TT stack(SparseVector b, int axis);


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
     * @param b Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     * entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    WW stack(CVector b, int axis);


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
     * @param b Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     * entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    WW stack(SparseCVector b, int axis);


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    U add(Vector B);


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    T add(SparseVector B);


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    CVector add(CVector B);


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    W add(SparseCVector B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    U sub(Vector B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    U sub(SparseVector B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    W sub(CVector B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     * @param B Vector to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    CVector sub(SparseCVector B);


    /**
     * Computes the element-wise addition between this vector and the specified vector and stores the result
     * in this vector.
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    void addEq(SparseVector B);


    /**
     * Computes the element-wise addition between this vector and the specified vector and stores the result
     * in this vector.
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    void addEq(Vector B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    void subEq(Vector B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     * @param B Vector to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    void subEq(SparseVector B);


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    T elemMult(Vector B);


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    V elemMult(SparseVector B);


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    W elemMult(CVector B);


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     * @param B Vector to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    SparseCVector elemMult(SparseCVector B);


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     * @param B Vector to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    T elemDiv(Vector B);


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     * @param B Vector to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    W elemDiv(CVector B);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    X inner(Vector b);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    X inner(SparseVector b);


    /**
     * Computes a unit vector in the same direction as this vector.
     * @return A unit vector with the same direction as this vector.
     */
    T normalize();


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    CNumber inner(CVector b);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    CNumber inner(SparseCVector b);


    /**
     * Computes the vector cross product between two vectors.
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    U cross(Vector b);


    /**
     * Computes the vector cross product between two vectors.
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    CVector cross(CVector b);


    /**
     * Computes the outer product of two vectors.
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    UU outer(Vector b);


    /**
     * Computes the outer product of two vectors.
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    UU outer(SparseVector b);


    /**
     * Computes the outer product of two vectors.
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    CMatrix outer(CVector b);


    /**
     * Computes the outer product of two vectors.
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    CMatrix outer(SparseCVector b);


    // TODO: ADD isParallel(CVector b), isParallel(SparseVector b), and isParallel(sparseCVector)
    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     */
    boolean isParallel(Vector b);


    // TODO: Add isPerp(CVector b), isPerp(SparseVector b), and isPerp(sparseCVector)
    /**
     * Checks if a vector is perpendicular to this vector.
     *
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     */
    boolean isPerp(Vector b);


    // TODO: Add toTensor methods.

    /**
     * Converts a vector to an equivalent matrix.
     * @return A matrix equivalent to this vector. This method will respect the orientation of the vector. That is, if
     * this vector is a row vector, then the resulting matrix will have a single row. If this vector is a column vector, then the
     * resulting matrix will have a single column.
     */
    TT toMatrix();


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     * @param columVector Flag for choosing whether to convert this vector to a matrix representing a row or column vector.
     *                    <p>If true, the vector will be converted to a matrix representing a column vector.</p>
     *                    <p>If false, The vector will be converted to a matrix representing a row vector.</p>
     * @return A matrix equivalent to this vector.
     */
    TT toMatrix(boolean columVector);
}
