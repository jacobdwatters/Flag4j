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

/**
 * This interface specifies methods which provide properties of a matrix. All matrices should implement this interface.
 *
 * @param <T> Matrix type.
 * @param <U> Dense Matrix type.
 * @param <V> Sparse Matrix type.
 * @param <W> Complex Matrix type.
 * @param <Y> Real Matrix type.
 * @param <X> Matrix entry type.
 */
public interface MatrixPropertiesMixin {


    /**
     * Checks if this matrix is square.
     * @return True if the matrix is square (i.e. the number of rows equals the number of columns). Otherwise, returns false.
     */
    boolean isSquare();


    /**
     * Checks if a matrix can be represented as a vector. That is, if a matrix has only one row or one column.
     * @return True if this matrix can be represented as either a row or column vector.
     */
    boolean isVector();


    /**
     * Checks what type of vector this matrix is. i.e. not a vector, a 1x1 matrix, a row vector, or a column vector.
     * @return - If this matrix can not be represented as a vector, then returns -1. <br>
     * - If this matrix is a 1x1 matrix, then returns 0. <br>
     * - If this matrix is a row vector, then returns 1. <br>
     * - If this matrix is a column vector, then returns 2.
     */
    int vectorType();


    /**
     * Checks if this matrix is triangular (i.e. upper triangular, diagonal, lower triangular).
     * @return True is this matrix is triangular. Otherwise, returns false.
     */
    boolean isTri();


    /**
     * Checks if this matrix is lower triangular.
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     */
    boolean isTriL();


    /**
     * Checks if this matrix is upper triangular.
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     */
    boolean isTriU();


    /**
     * Checks if this matrix is diagonal.
     * @return True is this matrix is diagonal. Otherwise, returns false.
     */
    boolean isDiag();


    /**
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    boolean isFullRank();


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.<br>
     * Also see {@link #isInvertible()}.
     * @return True if this matrix is singular. Otherwise, returns false.
     */
    boolean isSingular();


    /**
     * Checks if a matrix is invertible.<br>
     * Also see {@link #isSingular()}.
     * @return True if this matrix is invertible.
     */
    boolean isInvertible();


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    double norm(double p, double q);


    /**
     * Computes the max norm of a matrix.
     * @return The max norm of this matrix.
     */
    double maxNorm();


    /**
     * Computes the rank of this matrix (i.e. the dimension of the column space of this matrix).
     * Note that here, rank is <b>NOT</b> the same as a tensor rank.
     *
     * @return The matrix rank of this matrix.
     */
    int matrixRank();
}
