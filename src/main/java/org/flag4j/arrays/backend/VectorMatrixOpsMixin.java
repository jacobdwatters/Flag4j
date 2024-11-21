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

package org.flag4j.arrays.backend;

/**
 * This interface specifies operations between a vector and a matrix that a vector should implement.
 * @param <T> Type of the vector.
 * @param <U> Type of the matrix.
 * @param <V> Type of dense matrix equivalent to {@code U}.
 */
public interface VectorMatrixOpsMixin<T extends VectorMixinOld, U extends MatrixMixinOld, V extends MatrixMixinOld> {


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param n Number of times to repeat vector.
     * @param axis Axis along which to repeat vector:
     * <ul>
     *     <li>If {@code axis=0}, then the vector will be treated as a row vector and stacked vertically {@code n} times.</li>
     *     <li>If {@code axis=1} then the vector will be treated as a column vector and stacked horizontally {@code n} times.</li>
     * </ul>
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    public U repeat(int n, int axis);


    /**
     * Stacks two vectors vertically as if they were row vectors to form a matrix with two rows.
     *
     * @param b Vector to stack below this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of entries in
     *                                  the vector {@code b}.
     */
    public U stack(T b);


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
     * @param b    VectorOld to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of entries in this vector is different from the number of
     *                                  entries in the vector {@code b}.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    public U stack(T b, int axis);


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and {@code b}.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    public V outer(T b);


    /**
     * Converts a vector to an equivalent matrix representing the vector as a column vector.
     *
     * @return A matrix equivalent to this vector as if it were a column vector.
     */
    public default U toMatrix() {
        return toMatrix(true);
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     *                    <p>If {@code true}, the vector will be converted to a matrix representing a column vector.</p>
     *                    <p>If {@code false}, The vector will be converted to a matrix representing a row vector.</p>
     * @return A matrix equivalent to this vector.
     */
    public U toMatrix(boolean columVector);
}
