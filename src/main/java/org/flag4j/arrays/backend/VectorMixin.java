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
 * This interface specifies operations that a vector should implement.
 * @param <T> Type of the vector.
 * @param <U> Type of matrix equivalent to {@code T}.
 * @param <V> Type of dense matrix equivalent to {@code U}.
 * @param <W> Type (or wrapper of) an element of the vector.
 */
public interface VectorMixin<T extends VectorMixin<T, U, V, W>, U extends MatrixMixin<U, ?, W>,
        V extends MatrixMixin<V, ?, W>, W>
        extends TensorPropertiesMixin<W>, VectorMatrixOpsMixin<T, U, V> {

    /**
     * Joints specified vector with this vector. That is, creates a vector of length {@code this.length() + b.length()} containing
     * first the elements of this vector followed by the elements of {@code b}.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    public T join(T b);


    /**
     * <p>Computes the inner product between two vectors.</p>
     *
     * <p>Note: this method is distinct from {@link #dot(VectorMixin)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.</p>
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector {@code b}.
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     * @see #dot(VectorMixin)
     */
    public W inner(T b);


    /**
     * <p>Computes the dot product between two vectors.</p>
     *
     * <p>Note: this method is distinct from {@link #inner(VectorMixin)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.</p>
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of entries.
     * @see #inner(VectorMixin)
     */
    public W dot(T b);


    /**
     * Computes the Euclidean norm of this vector.
     * @return The Euclidean norm of this vector.
     */
    public double norm();


    /**
     * Computes the p-norm of this vector.
     * @param p {@code p} value in the p-norm.
     * @return The Euclidean norm of this vector.
     */
    public double norm(int p);


    /**
     * Computes a unit vector in the same direction as this vector.
     *
     * @return A unit vector with the same direction as this vector. If this vector is zeros, then an equivalently sized
     * zero vector will be returned.
     */
    public T normalize();


    /**
     * Checks if a vector is parallel to this vector.
     *
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is parallel to this vector and the same size. Otherwise, returns false.
     * @see #isPerp(VectorMixin)
     */
    public boolean isParallel(T b);


    /**
     * Checks if a vector is perpendicular to this vector.
     *
     * @param b Vector to compare to this vector.
     * @return True if the vector {@code b} is perpendicular to this vector and the same size. Otherwise, returns false.
     * @see #isParallel(VectorMixin)
     */
    public boolean isPerp(T b);


    /**
     * Gets the length of a vector.
     *
     * @return The length, i.e. the number of entries, in this vector.
     */
    public int length();
}
