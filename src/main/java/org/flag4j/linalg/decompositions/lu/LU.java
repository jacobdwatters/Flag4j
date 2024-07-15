/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.linalg.decompositions.lu;

import org.flag4j.arrays.sparse.PermutationMatrix;
import org.flag4j.core.MatrixMixin;
import org.flag4j.linalg.decompositions.Decomposition;
import org.flag4j.util.ArrayUtils;

/**
 * <p>This abstract class specifies methods for computing the LU decomposition of a matrix.</p>
 * <p>The {@code LU} decomposition, decomposes a matrix {@code A} into a unit lower triangular matrix {@code L}
 * and an upper triangular matrix {@code U} such that {@code A=LU}.</p>
 * <p>If partial pivoting is used, the decomposition will also yield a permutation matrix {@code P} such that
 * {@code PA=LU}.</p>
 * <p>If full pivoting is used, the decomposition will yield an additional permutation matrix {@code Q} such that
 *  {@code PAQ=LU}.</p>
 */
public abstract class LU<T extends MatrixMixin<T, ?, ?, ?, ?, ?, ?>> implements Decomposition<T> {

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
    protected PermutationMatrix P;
    /**
     * Permutation matrix to store column swaps if full pivoting is used.
     */
    protected PermutationMatrix Q;

    protected int numRowSwaps; // Tracks the number of row swaps made during full/partial pivoting.
    protected int numColSwaps; // Tracks the number of column swaps made during full pivoting.
    protected int[] rowSwaps; // Array for keeping track of row swaps made with full/partial pivoting.
    protected int[] colSwaps; // Array for keeping track of column swaps made during full pivoting.


    /**
     * Constructs a LU decomposer to decompose the specified matrix.
     * @param pivoting Pivoting to use. If pivoting is 2, full pivoting will be used. If pivoting is 1, partial pivoting
     *                 will be used. If pivoting is any other value, no pivoting will be used.
     */
    public LU(int pivoting) {
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
    public LU(int pivoting, double zeroPivotTol) {
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
    public org.flag4j.linalg.decompositions.lu.LU<T> decompose(T src) {
        initLU(src);
        numRowSwaps = 0; // Set the number of row swaps to zero.
        numColSwaps = 0; // Set the number of row swaps to zero.

        if(pivotFlag==Pivoting.NONE) {
            rowSwaps = colSwaps = null;
            noPivot(); // Compute with no pivoting.
        } else if(pivotFlag==Pivoting.PARTIAL) {
            rowSwaps = ArrayUtils.intRange(0, LU.numRows());
            partialPivot();
        } else {
            rowSwaps = ArrayUtils.intRange(0, LU.numRows());
            colSwaps = ArrayUtils.intRange(0, LU.numCols());
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
     * Gets the {@code L} and {@code U} matrices of the decomposition combined in a single matrix.
     * @return The {@code L} and {@code U} matrices of the decomposition stored together in a single matrix.
     * The diagonal of {@code L} is all ones and is not stored allowing the diagonal of {@code U} to be stored along
     * the diagonal of the combined matrix.
     */
    public T getLU() {
        return LU;
    }


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
    public PermutationMatrix getP() {
        if(rowSwaps != null) {
            P = new PermutationMatrix(rowSwaps.clone());
        } else {
            P = null;
        }

        return P;
    }


    /**
     * Gets the column permutation matrix of the decomposition.
     * @return The column permutation matrix of the decomposition. If full pivoting was not used, null will be returned.
     */
    public PermutationMatrix getQ() {
        if(colSwaps != null) {
            Q = new PermutationMatrix(colSwaps.clone()).inv(); // Invert to ensure matrix represents column swaps.
        } else {
            Q = null;
        }

        return Q;
    }


    /**
     * Gets the number of row swaps used in the last decomposition.
     * @return The number of row swaps used in the last decomposition.
     */
    public int getNumRowSwaps() {
        return numRowSwaps;
    }


    /**
     * Gets the number of column swaps used in the last decomposition.
     * @return The number of column swaps used in the last decomposition.
     */
    public int getNumColSwaps() {
        return numColSwaps;
    }


    /**
     * Tracks the swapping of two rows during gaussian elimination with partial or full pivoting.
     * @param rowIdx1 First row index in swap.
     * @param rowIdx2 Second row index in swap.
     */
    protected void swapRows(int rowIdx1, int rowIdx2) {
        numRowSwaps++;
        int temp = rowSwaps[rowIdx1];
        rowSwaps[rowIdx1] = rowSwaps[rowIdx2];
        rowSwaps[rowIdx2] = temp;

        LU.swapRows(rowIdx1, rowIdx2);
    }


    /**
     * Tracks the swapping of two columns during gaussian elimination with full pivoting.
     * @param colIdx1 First column index in swap.
     * @param colIdx2 Second column index in swap.
     */
    protected void swapCols(int colIdx1, int colIdx2) {
        numColSwaps++;
        int temp = colSwaps[colIdx1];
        colSwaps[colIdx1] = colSwaps[colIdx2];
        colSwaps[colIdx2] = temp;

        LU.swapCols(colIdx1, colIdx2);
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
