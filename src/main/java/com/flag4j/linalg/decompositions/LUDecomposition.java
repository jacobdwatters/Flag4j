/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.linalg.decompositions;


import com.flag4j.core.MatrixBase;

/**
 * <p>This abstract class specified methods for computing the LU decomposition of a matrix.</p>
 * <p>The {@code LU} decomposition, decomposes a matrix {@code A} into a unit lower triangular matrix {@code L}
 * and an upper triangular matrix {@code U} such that {@code A=LU}.</p>
 * <p>If partial pivoting is used, the decomposition will also yield a permutation matrix {@code P} such that
 * {@code PA=LU}.</p>
 * <p>If full pivoting is used, the decomposition will yield an additional permutation matrix {@code Q} such that
 *  {@code PAQ=LU}.</p>
 */
public abstract class LUDecomposition<T extends MatrixBase> implements Decomposition<T> {
    /**
     * Flag indicating what pivoting to use.
     */
    final Pivoting pivotFlag;
    /**
     * for determining if pivot value is to be considered zero in LU decomposition with no pivoting.
     */
    final double DEFAULT_ZERO_PIVOT_TOLERANCE = 0.5e-14;
    /**
     * Tolerance for determining if pivot value is to be considered zero in LU decomposition with no pivoting.
     */
    double zeroPivotTol;


    /**
     * Constructs a LU decomposer to decompose the specified matrix.
     * @param pivoting Pivoting to use. If pivoting is 2, full pivoting will be used. If pivoting is 1, partial pivoting
     *                 will be used. If pivoting is any other value, no pivoting will be used.
     */
    public LUDecomposition(int pivoting) {
        pivotFlag = Pivoting.get(pivoting);
        zeroPivotTol = DEFAULT_ZERO_PIVOT_TOLERANCE;
    }


    /**
     * Constructs a LU decomposer to decompose the specified matrix.
     * @param pivoting Pivoting to use. If pivoting is 2, full pivoting will be used. If pivoting is 1, partial pivoting
     *                 will be used. If pivoting is any other value, no pivoting will be used.
     */
    public LUDecomposition(int pivoting, double zeroPivotTol) {
        pivotFlag = Pivoting.get(pivoting);
        this.zeroPivotTol = zeroPivotTol;
    }


    /**
     * Storage for L and U matrices. Stored in a single matrix
     */
    protected T LU;
    /**
     * Permutation matrix to store row swaps if partial pivoting is used.
     */
    protected T P;
    /**
     * Permutation matrix to store row swaps if partial pivoting is used.
     */
    protected T Q;


    /**
     * Gets the unit lower triangular matrix of the decomposition.
     * @return The unit lower triangular matrix of the decomposition.
     */
    public abstract T getL();


    /**
     * Gets the upper triangular matrix of the decomposition.
     * @return The upper triangular matrix of the decomposition.
     */
    public abstract T getU();


    /**
     * Gets the row permutation matrix of the decomposition.
     * @return The row permutation matrix of the decomposition. If no pivoting was used, null will be returned.
     */
    public abstract T getP();


    /**
     * Gets the column permutation matrix of the decomposition.
     * @return The column permutation matrix of the decomposition. If full pivoting was not used, null will be returned.
     */
    public abstract T getQ();


    /**
     * Simple enum containing pivoting options for pivoting in LU decomposition.
     */
    public enum Pivoting {
        NONE, PARTIAL, FULL;

        /**
         * Converts an ordinal to the corresponding pivot flag.
         * @param ordinal Ordinal to convert.
         * @return The pivot flag with the corresponding ordinal. If the no pivot exists with the specified ordinal,
         * {@link #PARTIAL} is returned.
         */
        private static Pivoting get(int ordinal) {
            if(ordinal==Pivoting.FULL.ordinal()) {
                return Pivoting.FULL;
            } else if(ordinal==Pivoting.NONE.ordinal()) {
                return Pivoting.NONE;
            } else {
                return Pivoting.PARTIAL;
            }
        }
    }
}
