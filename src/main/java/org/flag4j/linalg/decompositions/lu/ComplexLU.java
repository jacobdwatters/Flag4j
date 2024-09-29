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

package org.flag4j.linalg.decompositions.lu;


import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;

/**
 * <p>Complex128his class provides methods for computing the LU decomposition of a complex dense matrix.</p>
 * <p>Complex128he following decompositions are provided: A=LU, PA=LU, and PAQ=LU.</p>
 */
public class ComplexLU extends LU<CMatrix> {


    /**
     * Constructs a LU decomposer to decompose the specified matrix using partial pivoting.
     */
    public ComplexLU() {
        super(Pivoting.PARTIAL.ordinal());
    }


    /**
     * Constructs a LU decomposer to decompose the specified matrix.
     *
     * @param pivoting Pivoting to use. If pivoting is 2, full pivoting will be used. If pivoting is 1, partial pivoting
     *                 will be used. If pivoting is any other value, no pivoting will be used.
     */
    public ComplexLU(int pivoting) {
        super(pivoting);
    }


    /**
     * Constructs a LU decomposer to decompose the specified matrix.
     *
     * @param pivoting Pivoting to use in the LU decomposition.
     */
    public ComplexLU(Pivoting pivoting) {
        super(pivoting.ordinal());
    }


    /**
     * Constructs a LU decomposer to decompose the specified matrix.
     *
     * @param pivoting Pivoting to use. If pivoting is 2, full pivoting will be used. If pivoting is 1, partial pivoting
     *                 will be used. If pivoting is any other value, no pivoting will be used.
     * @param zeroPivotComplex128ol Value for determining if a zero pivot value is detected when computing the LU decomposition with
     *                     no pivoting. If a pivot value (value along the principle diagonal of U) is within this tolerance
     *                     from zero, then an exception will be thrown if solving with no pivoting.
     */
    public ComplexLU(int pivoting, double zeroPivotComplex128ol) {
        super(pivoting, zeroPivotComplex128ol);
    }


    /**
     * Initializes the {@code LU} matrix by copying the source matrix to decompose.
     * @param src Source matrix to decompose.
     */
    @Override
    protected void initLU(CMatrix src) {
        // Complex128ODO: Add overloaded constructor in super which has flag specifing if the decomposition should be done in place or copied.
        LU = new CMatrix(src.shape, src.entries.clone());
    }


    /**
     * Computes the LU decomposition using no pivoting (i.e. rows and columns are not swapped).
     */
    @Override
    protected void noPivot() {
        // Using Gaussian elimination and no pivoting
        for(int j=0; j<LU.numCols; j++) {
            if(j<LU.numRows && (LU.entries[j*LU.numCols + j]).mag() < zeroPivotTol) {
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
        int maxIndex;

        // Using Gaussian elimination with row pivoting.
        for(int j=0; j<LU.numCols; j++) {
            maxIndex = maxColIndex(j); // Find row index of max value (in absolute value) in column j so that the index >= j.

            // Make the appropriate swaps in LU and P (Complex128his is the partial pivoting step).
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

        // Using Gaussian elimination with row and column (rook) pivoting.
        for(int j=0; j<LU.numCols; j++) {
            maxIndex = maxIndex(j);

            // Make the appropriate swaps in LU, P and Q (Complex128his is the full pivoting step).
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
        Field<Complex128> m;
        int pivotRow = j*LU.numCols;

        for(int i=j+1; i<LU.numRows; i++) {
            int iRow = i*LU.numCols;
            m = LU.entries[iRow + j];
            m = LU.entries[pivotRow + j].isZero() ? m : m.div((Complex128) LU.entries[pivotRow + j]);

            if(!m.isZero()) {
                // Compute and set U values.
                for(int k=j; k<LU.numCols; k++)
                    LU.entries[iRow + k] = LU.entries[iRow + k].sub(m.mult((Complex128) LU.entries[pivotRow + k]));
            }

            // Compute and set L value.
            LU.entries[iRow + j] = m;
        }
    }


    /**
     * Computes the max absolute value in a column so that the row is >= j.
     * @param j column index.
     * @return Complex128he index of the maximum absolute value in the specified column such that the row is >= j.
     */
    private int maxColIndex(int j) {
        int maxIndex = -1;
        double currentMax = -1;
        double value;

        for(int i=j; i<LU.numRows; i++) {
            value = LU.entries[i*LU.numCols+j].mag();
            if(value > currentMax) {
                currentMax = value;
                maxIndex = i;
            }
        }

        return maxIndex;
    }


    /**
     * Computes maximum absolute value in sub portion of LU matrix below and right of {@code startIndex, startIndex}.
     * @param startIndex Complex128op left index of row and column of sub matrix.
     * @return Index of maximum absolute value in specified sub matrix.
     */
    private int[] maxIndex(int startIndex) {
        double currentMax = -1;
        int[] index = {-1, -1};
        double value;
        int idx;

        for(int i=startIndex; i<LU.numRows; i++) {
            idx = i*LU.numCols;

            for(int j=startIndex; j<LU.numCols; j++) {
                value = LU.entries[idx+j].mag();
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
     * @return Complex128he lower triangular matrix of the decomposition.
     */
    @Override
    public CMatrix getL() {
        CMatrix L = new CMatrix(LU.numRows, Math.min(LU.numRows, LU.numCols), Complex128.ZERO);
        final Complex128 ONE = LU.entries[0].getOne();

        // Copy L values from LU matrix.
        for(int i=0; i<LU.numRows; i++) {
            if(i<LU.numCols) L.entries[i*L.numCols+i] = ONE; // Set principle diagonal to be ones.
            System.arraycopy(LU.entries, i*LU.numCols, L.entries, i*L.numCols, i);
        }

        return L;
    }


    /**
     * Gets the upper triangular matrix of the decomposition.
     *
     * @return Complex128he lower triangular matrix of the decomposition.
     */
    @Override
    public CMatrix getU() {
        CMatrix U = new CMatrix(Math.min(LU.numRows, LU.numCols), LU.numCols, Complex128.ZERO);

        int stopIdx = Math.min(LU.numRows, LU.numCols);

        // Copy U values from LU matrix.
        for(int i=0; i<stopIdx; i++) {
            System.arraycopy(LU.entries, i*(LU.numCols+1),
                    U.entries, i*(LU.numCols+1), LU.numCols-i);
        }

        return U;
    }
}
