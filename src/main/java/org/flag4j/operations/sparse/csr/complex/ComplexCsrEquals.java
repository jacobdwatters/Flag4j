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

package org.flag4j.operations.sparse.csr.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.operations.common.complex.ComplexProperties;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class contains methods to check equality or approximate equality between two complex sparse CSR matrices.
 */
public class ComplexCsrEquals {

    private ComplexCsrEquals() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     * @param src1 First matrix in the comparison.
     * @param src2 Second matrix in the comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if the {@code src1} matrix is the same shape as the {@code src2} matrix and all entries
     * are 'close', i.e. elements {@code a} and {@code b} at the same positions in the two matrices respectively
     * satisfy {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns false.
     */
    public static boolean allClose(CsrCMatrix src1, CsrCMatrix src2, double relTol, double absTol) {
        boolean close = src1.shape.equals(src2.shape);

        if(close) {
            // Remove values which are 'close' to zero.
            List<Complex128> src1Entries = new ArrayList<>(src1.entries.length);
            List<Integer> src1ColIndices = new ArrayList<>(src1Entries.size());
            int[] src1RowPointers = new int[src1.rowPointers.length];
            removeCloseToZero(src1, src1Entries, src1RowPointers, src1ColIndices, absTol);

            List<Complex128> src2Entries = new ArrayList<>(src2.entries.length);
            List<Integer> src2ColIndices = new ArrayList<>(src2Entries.size());
            int[] src2RowPointers = new int[src2.rowPointers.length];
            removeCloseToZero(src2, src2Entries, src2RowPointers, src2ColIndices, absTol);

            close = Arrays.equals(src1RowPointers, src2RowPointers)

                    && Arrays.equals(ArrayUtils.fromIntegerList(src1ColIndices),
                    ArrayUtils.fromIntegerList(src2ColIndices))

                    && ComplexProperties.allClose(ArrayUtils.fromList(src1Entries, new Complex128[src1Entries.size()]),
                    ArrayUtils.fromList(src2Entries, new Complex128[src2Entries.size()]), relTol, absTol);
        }

        return close;
    }


    /**
     * Removes entries in {@code src} which are within {@code atol} in absolute value from zero.
     * @param src Source CSR matrix.
     * @param entries List to store value in {@code src} which are not within {@code atol} in absolute value from zero.
     * @param colIndices Column indices of entries.
     * @param rowPointers Row pointers for entries.
     * @param aTol Absolute tolerance for value to be considered close to zero.
     */
    private static void removeCloseToZero(CsrCMatrix src, List<Complex128> entries, int[] rowPointers,
                                          List<Integer> colIndices, double aTol) {
        for(int i=0; i<src.numRows; i++) {
            int start = src.rowPointers[i];
            int stop = src.rowPointers[i+1];

            for(int j=start; j<stop; j++) {
                Complex128 value = (Complex128) src.entries[j];

                if(value.abs() > aTol) {
                    // Then keep value.
                    entries.add(value);
                    colIndices.add(src.colIndices[j]);
                    rowPointers[i]++;
                }
            }
        }

        // Accumulate row pointers.
        int size = rowPointers.length-1;
        for(int i=0; i<size; i++) {
            rowPointers[i+1] += rowPointers[i];
        }
    }
}
