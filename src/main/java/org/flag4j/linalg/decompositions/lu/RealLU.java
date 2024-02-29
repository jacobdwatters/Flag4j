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

import org.flag4j.dense.Matrix;
import org.flag4j.util.exceptions.LinearAlgebraException;

/**
 * <p>This class provides methods for computing the LU decomposition of a real dense matrix.</p>
 * <p>The following decompositions are provided: {@code A=LU}, {@code PA=LU}, and {@code PAQ=LU}.</p>
 */
public final class RealLU extends org.flag4j.linalg.decompositions.lu.LU<Matrix> {

    /**
     * Constructs a LU decomposer to decompose the specified matrix using partial pivoting.
     */
    public RealLU() {
        super(Pivoting.PARTIAL.ordinal());
    }


    /**
     * Constructs a LU decomposer to decompose the specified matrix.
     *
     * @param pivoting Pivoting to use. If pivoting is 2, full pivoting will be used. If pivoting is 1, partial pivoting
     *                 will be used. If pivoting is any other value, no pivoting will be used.
     */
    public RealLU(int pivoting) {
        super(pivoting);
    }


    /**
     * Constructs a LU decomposer to decompose the specified matrix.
     *
     * @param pivoting Pivoting to use. If pivoting is 2, full pivoting will be used. If pivoting is 1, partial pivoting
     *                 will be used. If pivoting is any other value, no pivoting will be used.
     * @param zeroPivotTol Value for determining if a zero pivot value is detected when computing the LU decomposition with
     *                     no pivoting. If a pivot value (value along the principle diagonal of U) is within this tolerance
     *                     from zero, then an exception will be thrown if solving with no pivoting.
     */
    public RealLU(int pivoting, double zeroPivotTol) {
        super(pivoting, zeroPivotTol);
    }


    /**
     * Initializes the {@code LU} matrix by copying the source matrix to decompose.
     * @param src Source matrix to decompose.
     */
    @Override
    protected void initLU(Matrix src) {
        LU = new Matrix(src);
    }


    /**
     * Computes the LU decomposition using no pivoting (i.e. rows and columns are not swapped).
     */
    @Override
    protected void noPivot() {
        int colStop = Math.min(LU.numCols, LU.numRows);

        // Using Gaussian elimination and no pivoting
        for(int j=0; j<colStop; j++) {
            if(j<LU.numRows && Math.abs(LU.entries[j*LU.numCols + j]) < zeroPivotTol) {
                throw new LinearAlgebraException("Zero pivot encountered in decomposition." +
                        " Consider using LU decomposition with partial pivoting.");
            }

            computeRows(j);
        }
    }


    /**
     * Computes the LU decomposition using partial pivoting (i.e. row swapping).
     */
    @Override
    protected void partialPivot() {
        int colStop = Math.min(LU.numCols, LU.numRows);
        int maxIndex;

        // Using Gaussian elimination with row pivoting.
        for(int j=0; j<colStop; j++) {
            maxIndex = maxColIndex(j); // Find row index of max value (in absolute value) in column j so that the index >= j.

            // Make the appropriate swaps in LU and P (This is the partial pivoting step).
            if(j!=maxIndex && maxIndex>=0) {
                swapRows(j, maxIndex);
            }

            computeRows(j);
        }
    }


    /**
     * Computes the LU decomposition using full/rook pivoting (i.e. row and column swapping).
     */
    @Override
    protected void fullPivot() {
        int[] maxIndex;
        int colStop = Math.min(LU.numCols, LU.numRows);

        // Using Gaussian elimination with row and column (rook) pivoting.
        for(int j=0; j<colStop; j++) {
            maxIndex = maxIndex(j);

            // Make the appropriate swaps in LU, P and Q (This is the full pivoting step).
            if(j!=maxIndex[0] && maxIndex[0]!=-1) {
                swapRows(j, maxIndex[0]);
            }
            if(j!=maxIndex[1] && maxIndex[1]!=-1) {
                swapCols(j, maxIndex[1]);
            }

            computeRows(j);
        }
    }


    /**
     * Helper method which computes rows in the gaussian elimination algorithm.
     * @param j Column for which to compute values to the right of.
     */
    private void computeRows(int j) {
        double m;
        int pivotRow = j*LU.numCols;
        int iRow;

        for(int i=j+1; i<LU.numRows; i++) {
            iRow = i*LU.numCols;
            m = LU.entries[iRow + j];
            m = LU.entries[pivotRow + j] == 0 ? m : m/LU.entries[pivotRow + j];

            if(m!=0) {
                // Compute and set U values.
                for (int k = j; k < LU.numCols; k++) {
                    LU.entries[iRow + k] -= m * LU.entries[pivotRow + k];
                }
            }

            // Compute and set L value.
            LU.entries[iRow + j] = m;
        }
    }


    /**
     * Computes the max absolute value in a column so that the row is >= j.
     * @param j column index.
     * @return The index of the maximum absolute value in the specified column such that the row is >= j.
     */
    private int maxColIndex(int j) {
        int maxIndex = -1;
        double currentMax = -1;
        double value;

        for(int i=j; i<LU.numRows; i++) {
            value = Math.abs(LU.entries[i*LU.numCols + j]);
            if(value > currentMax) {
                currentMax = value;
                maxIndex = i;
            }
        }

        return maxIndex;
    }


    /**
     * Computes maximum absolute value in sub portion of LU matrix below and right of {@code startIndex, startIndex}.
     * @param startIndex Top left index of row and column of sub matrix.
     * @return Index of maximum absolute value in specified sub matrix.
     */
    private int[] maxIndex(int startIndex) {
        double currentMax = -1;
        double value;
        int rowIdx;
        int[] index = {-1, -1};


        for(int i=startIndex; i<LU.numRows; i++) {
            rowIdx = i*LU.numCols;

            for(int j=startIndex; j<LU.numCols; j++) {
                value = Math.abs(LU.entries[rowIdx+j]);

                if(value > currentMax) {
                    currentMax = value;
                    index[0] = i;
                    index[1] = j;
                }
            }
        }

        return index;
    }


    /**
     * Gets the unit lower triangular matrix of the decomposition.
     *
     * @return The lower triangular matrix of the decomposition.
     */
    @Override
    public Matrix getL() {
        Matrix L = new Matrix(LU.numRows, Math.min(LU.numRows, LU.numCols));

        // Copy L values from LU matrix.
        for(int i=0; i<LU.numRows; i++) {
            if(i<LU.numCols) {
                L.entries[i*L.numCols+i] = 1; // Set principle diagonal to be 1, so it is unit lower-triangular.
            }
            System.arraycopy(LU.entries, i*LU.numCols, L.entries, i*L.numCols, i);
        }

        return L;
    }


    /**
     * Gets the upper triangular matrix of the decomposition.
     *
     * @return The lower triangular matrix of the decomposition.
     */
    @Override
    public Matrix getU() {
        Matrix U = new Matrix(Math.min(LU.numRows, LU.numCols), LU.numCols);

        int stopIdx = Math.min(LU.numRows, LU.numCols);

        // Copy U values from LU matrix.
        for(int i=0; i<stopIdx; i++) {
            System.arraycopy(LU.entries, i*(LU.numCols+1),
                    U.entries, i*(LU.numCols+1), LU.numCols-i);
        }

        return U;
    }
}
