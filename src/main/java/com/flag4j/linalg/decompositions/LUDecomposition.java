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

import com.flag4j.Matrix;
import com.flag4j.core.MatrixMixin;

// TODO: Implement LDU decomposition.

/**
 * <p>This abstract class specifies methods for computing the LU decomposition of a matrix.</p>
 * <p>The {@code LU} decomposition, decomposes a matrix {@code A} into a unit lower triangular matrix {@code L}
 * and an upper triangular matrix {@code U} such that {@code A=LU}.</p>
 * <p>If partial pivoting is used, the decomposition will also yield a permutation matrix {@code P} such that
 * {@code PA=LU}.</p>
 * <p>If full pivoting is used, the decomposition will yield an additional permutation matrix {@code Q} such that
 *  {@code PAQ=LU}.</p>
 */
public abstract class LUDecomposition<T extends MatrixMixin<T, ?, ?, ?, ?, ?, ?>> implements Decomposition<T> {

    /**
     * Flag indicating what pivoting to use.
     */
    public final Pivoting pivotFlag;
    /**
     * for determining if pivot value is to be considered zero in LU decomposition with no pivoting.
     */
    final double DEFAULT_ZERO_PIVOT_TOLERANCE = 0.5e-14;
    /**
     * Tolerance for determining if pivot value is to be considered zero in LU decomposition with no pivoting.
     */
    final double zeroPivotTol;
    /**
     * Storage for L and U matrices. Stored in a single matrix
     */
    protected T LU;
    /**
     * Permutation matrix to store row swaps if partial pivoting is used.
     */
    protected Matrix P;
    /**
     * Permutation matrix to store column swaps if full pivoting is used.
     */
    protected Matrix Q;


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
     * @param zeroPivotTol Tolerance for considering a pivot to be zero. If a pivot is less than the tolerance in absolute value,
     *                     then it will be considered zero.
     */
    public LUDecomposition(int pivoting, double zeroPivotTol) {
        pivotFlag = Pivoting.get(pivoting);
        this.zeroPivotTol = zeroPivotTol;
    }


    /**
     * Applies {@code LU} decomposition to the source matrix using the pivoting specified in the constructor.
     *
     * @param src The source matrix to decompose. Not modified.
     * @return A reference to this decomposer.
     */
    @Override
    public LUDecomposition<T> decompose(T src) {
        initLU(src);

        if(pivotFlag==Pivoting.NONE) {
            noPivot(); // Compute with no pivoting.
        } else if(pivotFlag==Pivoting.PARTIAL) {
            partialPivot();
        } else {
            fullPivot();
        }

        return this;
    }


    /**
     * Initializes the {@code LU} matrix by copying the source matrix to decompose.
     * @param src Source matrix to decompose.
     */
    protected abstract void initLU(T src);


    /**
     * Computes the LU decomposition using no pivoting (i.e. rows and columns are not swapped).
     */
    protected abstract void noPivot();


    /**
     * Computes the LU decomposition using partial pivoting (i.e. row swapping).
     */
    protected abstract void partialPivot();


    /**
     * Computes the LU decomposition using full/rook pivoting (i.e. row and column swapping).
     */
    protected abstract void fullPivot();


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
    public Matrix getP() {
        return P;
    }


    /**
     * Gets the column permutation matrix of the decomposition.
     * @return The column permutation matrix of the decomposition. If full pivoting was not used, null will be returned.
     */
    public Matrix getQ() {
        return Q;
    }


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
