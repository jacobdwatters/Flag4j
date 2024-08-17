/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.core;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;

/**
 * This interface specifies operations_old which should be implemented by any vector.
 * @param <T> VectorOld type.
 * @param <U> Dense VectorOld type.
 * @param <V> Sparse VectorOld type.
 * @param <W> Complex VectorOld type.
 * @param <X> VectorOld entry type.
 * @param <TT> MatrixOld type equivalent.
 * @param <UU> Dense MatrixOld type equivalent.
 * @param <WW> Complex MatrixOld type equivalent.
 */
public interface VectorOperationsMixin<T, U, V, W, X extends Number, TT, UU, WW> {


    /**
     * Joints specified vector with this vector.
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    U join(VectorOld b);


    /**
     * Joints specified vector with this vector.
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    CVectorOld join(CVectorOld b);


    /**
     * Joints specified vector with this vector.
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    T join(CooVector b);


    /**
     * Joints specified vector with this vector.
     * @param b VectorOld to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    W join(CooCVector b);

    // TODO: Add stack(vec, axis) methods so vectors can be stacked as if column vectors.

    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     * - If the number of entries in this vector is different from the number of entries in
     * the vector {@code b}.
     */
    TT stack(VectorOld b);


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     * - If the number of entries in this vector is different from the number of entries in
     * the vector {@code b}.
     */
    TT stack(CooVector b);


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     * - If the number of entries in this vector is different from the number of entries in
     * the vector {@code b}.
     */
    WW stack(CVectorOld b);


    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param b VectorOld to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException <br>
     * - If the number of entries in this vector is different from the number of entries in
     * the vector {@code b}.
     */
    WW stack(CooCVector b);


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
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     * entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    TT stack(VectorOld b, int axis);


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
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     * entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    TT stack(CooVector b, int axis);


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
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     * entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    WW stack(CVectorOld b, int axis);


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
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     * entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    WW stack(CooCVector b, int axis);


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    U add(VectorOld B);


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    T add(CooVector B);


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    CVectorOld add(CVectorOld B);


    /**
     * Computes the element-wise addition between this vector and the specified vector.
     * @param B VectorOld to add to this vector.
     * @return The result of the element-wise vector addition.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    W add(CooCVector B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     * @param B VectorOld to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    U sub(VectorOld B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     * @param B VectorOld to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    T sub(CooVector B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     * @param B VectorOld to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    CVectorOld sub(CVectorOld B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector.
     * @param B VectorOld to subtract from this vector.
     * @return The result of the element-wise vector subtraction.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    W sub(CooCVector B);


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     * @param B VectorOld to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    T elemMult(VectorOld B);


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     * @param B VectorOld to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    V elemMult(CooVector B);


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     * @param B VectorOld to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    W elemMult(CVectorOld B);


    /**
     * Computes the element-wise multiplication (Hadamard multiplication) between this vector and a specified vector.
     * @param B VectorOld to element-wise multiply to this vector.
     * @return The vector resulting from the element-wise multiplication.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    CooCVector elemMult(CooCVector B);


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     * @param B VectorOld to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    T elemDiv(VectorOld B);


    /**
     * Computes the element-wise division (Hadamard multiplication) between this vector and a specified vector.
     * @param B VectorOld to element-wise divide this vector by.
     * @return The vector resulting from the element-wise division.
     * @throws IllegalArgumentException If this vector and {@code B} do not have the same size.
     */
    W elemDiv(CVectorOld B);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    X inner(VectorOld b);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    X inner(CooVector b);


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    T normalize();


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    CNumber inner(CVectorOld b);


    /**
     * Computes the inner product between two vectors.
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    CNumber inner(CooCVector b);


    /**
     * Computes the outer product of two vectors.
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    UU outer(VectorOld b);


    /**
     * Computes the outer product of two vectors.
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    UU outer(CooVector b);


    /**
     * Computes the outer product of two vectors.
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    CMatrixOld outer(CVectorOld b);


    /**
     * Computes the outer product of two vectors.
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    CMatrixOld outer(CooCVector b);


    // TODO: ADD isParallel(CVectorOld b), isParallel(CooVector b), and isParallel(sparseCVector)
    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b VectorOld to compare to this vector.
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     */
    boolean isParallel(VectorOld b);


    // TODO: Add isPerp(CVectorOld b), isPerp(CooVector b), and isPerp(sparseCVector)
    /**
     * Checks if a vector is perpendicular to this vector.
     *
     * @param b VectorOld to compare to this vector.
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     */
    boolean isPerp(VectorOld b);


    // TODO: Add toTensor methods.

    /**
     * Converts a vector to an equivalent matrix.
     * @return A matrix equivalent to this vector where the vector is treated as a column vector.
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
