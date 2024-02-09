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

package com.flag4j.operations.sparse.csr.real;

import com.flag4j.dense.Matrix;
import com.flag4j.sparse.CsrMatrix;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BinaryOperator;


/**
 * This class contains low-level implementations for element-wise operations on CSR matrices.
 */
public final class RealCsrOperations {

    private RealCsrOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Applies an element-wise binary operation to two {@link CsrMatrix CSR Matrices}. <br><br>
     *
     * Note, this methods efficiency relies heavily on the assumption that both operand matrices are very large and very
     * sparse. If the two matrices are not large and very sparse, this method will likely be
     * significantly slower than simply converting the matrices to {@link Matrix dense matrices} and using a dense
     * matrix addition algorithm.
     * @param src1 The first matrix in the operation.
     * @param src2 The second matrix in the operation.
     * @param opp Binary operator to apply element-wise to <code>src1</code> and <code>src2</code>.
     * @return The result of applying the specified binary operation to <code>src1</code> and <code>src2</code>
     * element-wise.
     * @throws IllegalArgumentException If <code>src1</code> and <code>src2</code> do not have the same shape.
     */
    public static CsrMatrix applyBinOpp(CsrMatrix src1, CsrMatrix src2, BinaryOperator<Double> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        List<Double> dest = new ArrayList<>();
        int[] rowPointers = new int[src1.rowPointers.length];
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<src1.numRows; i++) {
            int rowPtr1 = src1.rowPointers[i];
            int rowPtr2 = src2.rowPointers[i];

            while(rowPtr1 < src1.rowPointers[i+1] && rowPtr2 < src2.rowPointers[i+1]) {
                int col1 = src1.colIndices[rowPtr1];
                int col2 = src2.colIndices[rowPtr2];

                if(col1 == col2) {
                    dest.add(opp.apply(src1.entries[rowPtr1], src2.entries[rowPtr2]));
                    colIndices.add(col1);
                    rowPtr1++;
                    rowPtr2++;
                } else if(col1 < col2) {
                    dest.add(src1.entries[rowPtr1]);
                    colIndices.add(col1);
                    rowPtr1++;
                } else {
                    dest.add(src2.entries[rowPtr2]);
                    colIndices.add(col2);
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }

            while(rowPtr1 < src1.rowPointers[i+1]) {
                dest.add(src1.entries[rowPtr1]);
                colIndices.add(src1.colIndices[rowPtr1]);
                rowPtr1++;
                rowPointers[i+1]++;
            }

            while(rowPtr2 < src2.rowPointers[i+1]) {
                dest.add(src2.entries[rowPtr2]);
                colIndices.add(src2.colIndices[rowPtr2]);
                rowPtr2++;
                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrMatrix(src1.shape.copy(),
                ArrayUtils.fromDoubleList(dest),
                rowPointers,
                ArrayUtils.fromIntegerList(colIndices)
        );
    }


    /**
     * Transposes a sparse CSR matrix.
     * @param src The matrix to transpose.
     * @return The transpose of the {@code src} matrix.
     */
    public static CsrMatrix transpose(CsrMatrix src) {

        double[] dest = new double[src.entries.length];
        int[] rowPtrs = new int[src.numCols+1];
        int[] colIndices = new int[src.entries.length];

        // Count number of entries in each row.
        for(int i=0; i<src.colIndices.length; i++) {
            rowPtrs[src.colIndices[i] + 1]++;
        }

        // Accumulate the row counts.
        for(int i=1; i<rowPtrs.length; i++) {
            rowPtrs[i] += rowPtrs[i-1];
        }

        int[] tempPos = java.util.Arrays.copyOf(rowPtrs, rowPtrs.length);

        // Fill in the values for the transposed matrix
        for (int row = 0; row < src.numRows; row++) {
            for (int i = src.rowPointers[row]; i < src.rowPointers[row + 1]; i++) {
                int col = src.colIndices[i];
                int pos = tempPos[col];
                dest[pos] = src.entries[i];
                colIndices[pos] = row;
                tempPos[col]++;
            }
        }

        return new CsrMatrix(src.numCols, src.numRows, dest, rowPtrs, colIndices);
    }


    /**
     * Compute the L<sub>p,q</sub> norm of a sparse CSR matrix.
     * @param src Sparse CSR matrix to compute norm of.
     * @return The L<sub>p,q</sub> norm of the matrix.
     */
    public static double matrixNormLpq(CsrMatrix src, double p, double q) {
        CsrMatrix tSrc = transpose(src);
        double norm = 0;
        double pOverQ = p/q;

        for(int i=0; i<tSrc.numRows; i++) {
            int start = tSrc.rowPointers[i];
            int stop = tSrc.rowPointers[i+1];
            double colNorm = 0;

            for(int j=start; j<stop; j++) {
                colNorm += Math.pow(Math.abs(tSrc.entries[j]), p);
            }

            norm += Math.pow(colNorm, pOverQ);
        }

        return Math.pow(norm, 1.0/q);
    }
}





