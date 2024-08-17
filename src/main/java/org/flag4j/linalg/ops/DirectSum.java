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

package org.flag4j.linalg.ops;


import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooMatrix;
import org.flag4j.arrays_old.sparse.CsrCMatrix;
import org.flag4j.arrays_old.sparse.CsrMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;

/**
 * Utility class for computing the direct sum between two matrices.
 */
public final class DirectSum {

    private DirectSum() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }

    // ------------------------------ Real Dense MatrixOld ------------------------------
    /**
     * Computes the direct sum of two matrices.
     * 
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static MatrixOld directSum(MatrixOld A, MatrixOld B) {
        MatrixOld sum = new MatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols, sum.entries, i*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            System.arraycopy(B.entries, i*B.numCols, sum.entries, (i + A.numRows)*sum.numCols + A.numCols, B.numCols);
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     * 
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static MatrixOld directSum(MatrixOld A, CooMatrix B) {
        MatrixOld sum = new MatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols, sum.entries, i*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[(row+A.numRows)*sum.numCols + (col+A.numCols)] = B.entries[i];
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     * 
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld directSum(MatrixOld A, CMatrixOld B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            for(int j=0; j<A.numCols; j++) {
                sum.entries[i*sum.numCols + j] = new CNumber(A.entries[i*A.numCols + j]);
            }
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            System.arraycopy(B.entries, i*B.numCols, sum.entries, (i+A.numRows)*sum.numCols+(A.numCols), B.numCols);
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     * 
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld directSum(MatrixOld A, CooCMatrix B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            for(int j=0; j<A.numCols; j++) {
                sum.entries[i*sum.numCols + j] = new CNumber(A.entries[i*A.numCols + j]);
            }
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[(row+A.numRows)*sum.numCols + (col+A.numCols)] = B.entries[i];
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * 
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static MatrixOld invDirectSum(MatrixOld A, MatrixOld B) {
        MatrixOld sum = new MatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols, sum.entries, (i+B.numRows)*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            System.arraycopy(B.entries, i*B.numCols, sum.entries, i*sum.numCols+A.numCols, B.numCols);
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * 
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static MatrixOld invDirectSum(MatrixOld A, CooMatrix B) {
        MatrixOld sum = new MatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols, sum.entries, (i+B.numRows)*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[row*sum.numCols + col + A.numCols] = B.entries[i];
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     * 
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld invDirectSum(MatrixOld A, CMatrixOld B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            for(int j=0; j<A.numCols; j++) {
                sum.entries[(i+B.numRows)*sum.numCols + j] = new CNumber(A.entries[i*A.numCols + j]);
            }
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            for(int j=0; j<B.numCols; j++) {
                sum.entries[i*sum.numCols + j + A.numCols] = B.entries[i*B.numCols + j];
            }
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld invDirectSum(MatrixOld A, CooCMatrix B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            for(int j=0; j<A.numCols; j++) {
                sum.entries[(i+B.numRows)*sum.numCols + j] = new CNumber(A.entries[i*A.numCols + j]);
            }
        }

        // Copy over second matrix.
        int row, col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[row*sum.numCols + col + A.numCols] = B.entries[i];
        }

        return sum;
    }
    // -------------------------------------------------------------------------------


    // ---------------------------- Complex Dense MatrixOld -----------------------------
    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld directSum(CMatrixOld A, MatrixOld B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols , sum.entries, i*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            for(int j=0; j<B.numCols; j++) {
                sum.entries[(i+A.numRows)*sum.numCols + j + A.numCols] = new CNumber(B.entries[i*B.numCols + j]);
            }
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld directSum(CMatrixOld A, CooMatrix B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols, sum.entries, i*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        int row;
        int col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[(row+A.numRows)*sum.numCols + (col+A.numCols)] = new CNumber(B.entries[i]);
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld directSum(CMatrixOld A, CMatrixOld B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols, sum.entries, i*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            System.arraycopy(B.entries, i*B.numCols, sum.entries, (i+A.numRows)*sum.numCols + A.numCols, B.numCols);
        }

        return sum;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld directSum(CMatrixOld A, CooCMatrix B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols, sum.entries, i*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        int row;
        int col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[(row+A.numRows)*sum.numCols + (col+A.numCols)] = B.entries[i];
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld invDirectSum(CMatrixOld A, MatrixOld B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols, sum.entries, (i+B.numRows)*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            int sumRowOffset = i*sum.numCols + A.numCols;
            int bRowOffset = i*B.numCols;

            for(int j=0; j<B.numCols; j++) {
                sum.entries[sumRowOffset + j] = new CNumber(B.entries[bRowOffset + j]);
            }
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld invDirectSum(CMatrixOld A, CooMatrix B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            System.arraycopy(A.entries, i*A.numCols, sum.entries, (i+B.numRows)*sum.numCols, A.numCols);
        }

        // Copy over second matrix.
        int row;
        int col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[row*sum.numCols + col + A.numCols] = new CNumber(B.entries[i]);
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld invDirectSum(CMatrixOld A, CMatrixOld B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            int sumRowOffset = (i+B.numRows)*sum.numCols;
            int aRowOffset = i*A.numCols;

            for(int j=0; j<A.numCols; j++) {
                sum.entries[sumRowOffset + j] = A.entries[aRowOffset + j];
            }
        }

        // Copy over second matrix.
        for(int i=0; i<B.numRows; i++) {
            for(int j=0; j<B.numCols; j++) {
                sum.entries[i*sum.numCols + j + A.numCols] = B.entries[i*B.numCols + j];
            }
        }

        return sum;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CMatrixOld invDirectSum(CMatrixOld A, CooCMatrix B) {
        CMatrixOld sum = new CMatrixOld(A.numRows+B.numRows, A.numCols+B.numCols);

        // Copy over first matrix.
        for(int i=0; i<A.numRows; i++) {
            int sumRowOffset = (i+B.numRows)*sum.numCols;
            int aRowOffset = i*A.numCols;

            for(int j=0; j<A.numCols; j++) {
                sum.entries[sumRowOffset + j] = A.entries[aRowOffset + j];
            }
        }

        // Copy over second matrix.
        int row;
        int col;
        for(int i=0; i<B.nonZeroEntries(); i++) {
            row = B.rowIndices[i];
            col = B.colIndices[i];

            sum.entries[row*sum.numCols + col + A.numCols] = B.entries[i];
        }

        return sum;
    }
    // -------------------------------------------------------------------------------


    // ------------------------------- Real COO MatrixOld -------------------------------
    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooMatrix directSum(CooMatrix A, MatrixOld B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        double[] destEntries = new double[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(A.entries, 0, destEntries, 0, A.entries.length);
        System.arraycopy(B.entries, 0, destEntries, A.entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+A.entries.length;
            indices = B.shape.getIndices(i);
            destRowIndices[destIdx] = indices[0] + A.numRows;
            destColIndices[destIdx] = indices[1] + A.numCols;
        }

        return new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooMatrix directSum(CooMatrix A, CooMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        double[] destEntries = new double[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(A.entries, 0, destEntries, 0, A.entries.length);
        System.arraycopy(B.entries, 0, destEntries, A.entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+A.entries.length;
            destRowIndices[destIdx] = B.rowIndices[i] + A.numRows;
            destColIndices[destIdx] = B.colIndices[i] + A.numCols;
        }

        return new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooMatrix A, CMatrixOld B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        ArrayUtils.arraycopy(A.entries, 0, destEntries, 0, A.entries.length);
        System.arraycopy(B.entries, 0, destEntries, A.entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+A.entries.length;
            indices = B.shape.getIndices(i);
            destRowIndices[destIdx] = indices[0] + A.numRows;
            destColIndices[destIdx] = indices[1] + A.numCols;
        }

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooMatrix A, CooCMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        ArrayUtils.arraycopy(A.entries, 0, destEntries, 0, A.entries.length);
        System.arraycopy(B.entries, 0, destEntries, A.entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+A.entries.length;
            destRowIndices[destIdx] = B.rowIndices[i] + A.numRows;
            destColIndices[destIdx] = B.colIndices[i] + A.numCols;
        }

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooMatrix invDirectSum(CooMatrix A, MatrixOld B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        double[] destEntries = new double[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(B.entries, 0, destEntries, 0, B.entries.length);
        System.arraycopy(A.entries, 0, destEntries, B.entries.length, A.entries.length);

        // Compute shifted indices.
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());
        int[] bRowIndices = ArrayUtils.intRange(0, B.numRows, B.numCols);
        int[] bColIndices = ArrayUtils.repeat(B.numRows, ArrayUtils.intRange(A.numCols, A.numCols + B.numCols));

        // Copy shifted indices of both matrices.
        System.arraycopy(bRowIndices, 0, destRowIndices, 0, bRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, 0, bColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, bRowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, bColIndices.length, A.colIndices.length);

        return new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooMatrix invDirectSum(CooMatrix A, CooMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        double[] destEntries = new double[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(B.entries, 0, destEntries, 0, B.entries.length);
        System.arraycopy(A.entries, 0, destEntries, B.entries.length, A.entries.length);

        // Compute shifted indices.
        int[] shiftedColIndices = ArrayUtils.shift(A.numCols, B.colIndices.clone());
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());

        // Copy shifted indices of both matrices.
        System.arraycopy(B.rowIndices, 0, destRowIndices, 0, B.rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, B.rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, B.colIndices.length, A.colIndices.length);

        return new CooMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooMatrix A, CMatrixOld B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(B.entries, 0, destEntries, 0, B.entries.length);
        ArrayUtils.arraycopy(A.entries, 0, destEntries, B.entries.length, A.entries.length);

        // Compute shifted indices.
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());
        int[] bRowIndices = ArrayUtils.intRange(0, B.numRows, B.numCols);
        int[] bColIndices = ArrayUtils.repeat(B.numRows, ArrayUtils.intRange(A.numCols, A.numCols + B.numCols));

        // Copy shifted indices of both matrices.
        System.arraycopy(bRowIndices, 0, destRowIndices, 0, bRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, 0, bColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, bRowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, bColIndices.length, A.colIndices.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooMatrix A, CooCMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(B.entries, 0, destEntries, 0, B.entries.length);
        ArrayUtils.arraycopy(A.entries, 0, destEntries, B.entries.length, A.entries.length);

        // Compute shifted indices.
        int[] shiftedColIndices = ArrayUtils.shift(A.numCols, B.colIndices.clone());
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());

        // Copy shifted indices of both matrices.
        System.arraycopy(B.rowIndices, 0, destRowIndices, 0, B.rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, B.rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, B.colIndices.length, A.colIndices.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }
    // -------------------------------------------------------------------------------


    // ------------------------------ Complex COO MatrixOld -----------------------------
    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooCMatrix A, MatrixOld B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(A.entries, 0, destEntries, 0, A.entries.length);
        ArrayUtils.arraycopy(B.entries, 0, destEntries, A.entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+A.entries.length;
            indices = B.shape.getIndices(i);
            destRowIndices[destIdx] = indices[0] + A.numRows;
            destColIndices[destIdx] = indices[1] + A.numCols;
        }

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooCMatrix A, CooMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(A.entries, 0, destEntries, 0, A.entries.length);
        ArrayUtils.arraycopy(B.entries, 0, destEntries, A.entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+A.entries.length;
            destRowIndices[destIdx] = B.rowIndices[i] + A.numRows;
            destColIndices[destIdx] = B.colIndices[i] + A.numCols;
        }

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooCMatrix A, CMatrixOld B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(A.entries, 0, destEntries, 0, A.entries.length);
        System.arraycopy(B.entries, 0, destEntries, A.entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;
        int[] indices;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+A.entries.length;
            indices = B.shape.getIndices(i);
            destRowIndices[destIdx] = indices[0] + A.numRows;
            destColIndices[destIdx] = indices[1] + A.numCols;
        }

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public static CooCMatrix directSum(CooCMatrix A, CooCMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(A.entries, 0, destEntries, 0, A.entries.length);
        System.arraycopy(B.entries, 0, destEntries, A.entries.length, B.entries.length);

        // Copy indices of first matrix in the direct sum.
        System.arraycopy(A.rowIndices, 0, destRowIndices, 0, A.rowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, 0, A.colIndices.length);

        int destIdx;

        // Copy indices of second matrix with appropriate shifts.
        for(int i=0; i<B.entries.length; i++) {
            destIdx = i+A.entries.length;
            destRowIndices[destIdx] = B.rowIndices[i] + A.numRows;
            destColIndices[destIdx] = B.colIndices[i] + A.numCols;
        }

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooCMatrix A, MatrixOld B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        ArrayUtils.arraycopy(B.entries, 0, destEntries, 0, B.entries.length);
        System.arraycopy(A.entries, 0, destEntries, B.entries.length, A.entries.length);

        // Compute shifted indices.
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());
        int[] bRowIndices = ArrayUtils.intRange(0, B.numRows, B.numCols);
        int[] bColIndices = ArrayUtils.repeat(B.numRows, ArrayUtils.intRange(A.numCols, A.numCols + B.numCols));

        // Copy shifted indices of both matrices.
        System.arraycopy(bRowIndices, 0, destRowIndices, 0, bRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, 0, bColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, bRowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, bColIndices.length, A.colIndices.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooCMatrix A, CooCMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(B.entries, 0, destEntries, 0, B.entries.length);
        System.arraycopy(A.entries, 0, destEntries, B.entries.length, A.entries.length);

        // Compute shifted indices.
        int[] shiftedColIndices = ArrayUtils.shift(A.numCols, B.colIndices.clone());
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());

        // Copy shifted indices of both matrices.
        System.arraycopy(B.rowIndices, 0, destRowIndices, 0, B.rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, B.rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, B.colIndices.length, A.colIndices.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooCMatrix A, CooMatrix B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        ArrayUtils.arraycopy(B.entries, 0, destEntries, 0, B.entries.length);
        System.arraycopy(A.entries, 0, destEntries, B.entries.length, A.entries.length);

        // Compute shifted indices.
        int[] shiftedColIndices = ArrayUtils.shift(A.numCols, B.colIndices.clone());
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());

        // Copy shifted indices of both matrices.
        System.arraycopy(B.rowIndices, 0, destRowIndices, 0, B.rowIndices.length);
        System.arraycopy(shiftedColIndices, 0, destColIndices, 0, shiftedColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, B.rowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, B.colIndices.length, A.colIndices.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public static CooCMatrix invDirectSum(CooCMatrix A, CMatrixOld B) {
        Shape destShape = new Shape(A.numRows + B.numRows, A.numCols + B.numCols);
        CNumber[] destEntries = new CNumber[A.entries.length + B.entries.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy entries from both matrices.
        System.arraycopy(B.entries, 0, destEntries, 0, B.entries.length);
        System.arraycopy(A.entries, 0, destEntries, B.entries.length, A.entries.length);

        // Compute shifted indices.
        int[] shiftedRowIndices = ArrayUtils.shift(B.numRows, A.rowIndices.clone());
        int[] bRowIndices = ArrayUtils.intRange(0, B.numRows, B.numCols);
        int[] bColIndices = ArrayUtils.repeat(B.numRows, ArrayUtils.intRange(A.numCols, A.numCols + B.numCols));

        // Copy shifted indices of both matrices.
        System.arraycopy(bRowIndices, 0, destRowIndices, 0, bRowIndices.length);
        System.arraycopy(bColIndices, 0, destColIndices, 0, bColIndices.length);

        System.arraycopy(shiftedRowIndices, 0, destRowIndices, bRowIndices.length, shiftedRowIndices.length);
        System.arraycopy(A.colIndices, 0, destColIndices, bColIndices.length, A.colIndices.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }
    // -------------------------------------------------------------------------------


    // ------------------------------- Real CSR MatrixOld -------------------------------
    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CsrMatrix directSum(CsrMatrix A, MatrixOld B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CsrMatrix directSum(CsrMatrix A, CooMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CsrCMatrix directSum(CsrMatrix A, CMatrixOld B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing {@code A} with {@code B}.
     */
    public static CsrCMatrix directSum(CsrMatrix A, CooCMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CsrMatrix invDirectSum(CsrMatrix A, MatrixOld B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CsrMatrix invDirectSum(CsrMatrix A, CsrMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CsrMatrix invDirectSum(CsrMatrix A, CooMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CsrCMatrix invDirectSum(CsrMatrix A, CMatrixOld B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing {@code A} with {@code B}.
     */
    public static CsrCMatrix invDirectSum(CsrMatrix A, CooCMatrix B) {
        // TODO: Implementation
        return null;
    }
    // -------------------------------------------------------------------------------


    // ------------------------------ Complex CSR MatrixOld -----------------------------
    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public CsrMatrix directSum(CsrCMatrix A, MatrixOld B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public CsrMatrix directSum(CsrCMatrix A, CooMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public CsrCMatrix directSum(CsrCMatrix A, CMatrixOld B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param A First matrix in the direct sum.
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    public CsrCMatrix directSum(CsrCMatrix A, CooCMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public CsrMatrix invDirectSum(CsrCMatrix A, MatrixOld B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public CsrMatrix invDirectSum(CsrCMatrix A, CsrMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public CsrMatrix invDirectSum(CsrCMatrix A, CooMatrix B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public CsrCMatrix invDirectSum(CsrCMatrix A, CMatrixOld B) {
        // TODO: Implementation
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param A First matrix in the inverse direct sum.
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    public CsrCMatrix invDirectSum(CsrCMatrix A, CooCMatrix B) {
        // TODO: Implementation
        return null;
    }
    // -------------------------------------------------------------------------------
}
