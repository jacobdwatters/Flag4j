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


import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.Vector;

/**
 * This interface specifies comparisons which all vectors should implement.
 *
 * @param <T> Vector type.
 * @param <U> Dense vector type.
 * @param <V> Sparse vector type.
 * @param <W> Complex vector type.
 * @param <Y> Real vector type.
 * @param <X> Vector entry type.
 */
public interface VectorComparisonsMixin<T, U, V, W, Y, X extends Number> extends TensorComparisonsMixin<T, U, V, W, Y, X> {


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    boolean sameShape(Vector b);


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    boolean sameShape(SparseVector b);


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    boolean sameShape(CVector b);


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    boolean sameShape(SparseCVector b);


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    boolean sameSize(Vector b);


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    boolean sameSize(SparseVector b);


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    boolean sameSize(CVector b);


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    boolean sameSize(SparseCVector b);
}
