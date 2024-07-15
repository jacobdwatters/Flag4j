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

package org.flag4j.arrays.sparse;

import org.flag4j.util.ErrorMessages;

/**
 * Utility class for computations with sparse tensor, matrices, and vectors.
 */
final class SparseUtils {

    private SparseUtils() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * <p>Checks if two {@link CsrMatrix CSR Matrices} are equal considering the fact that one may explicitly store zeros at some
     * position that the other does not store.</p>
     *
     * <p>
     * If zeros are explicitly stored at some position in only one matrix, it will be checked that the
     * other matrix does not have a non-zero value at the same index. If it does, the matrices will not be equal. If no value is
     * stored for that index then it is assumed to be zero by definition of the CSR format and the equality check continues with the
     * remaining values.
     * </p>
     *
     * @param src1 First CSR matrix in the equality comparison.
     * @param src2 Second CSR matrix in the equality comparison.
     * @return True if all non-zero values stored in the two matrices are equal and occur at the same indices.
     */
    static boolean CSREquals(CsrMatrix src1, CsrMatrix src2) {
        if(src1.numRows != src2.numRows || src1.numCols != src2.numCols) {
            System.out.println("false return 1");
            return false;
        }

        // Compare row by row
        for (int i=0; i<src1.numRows; i++) {
            int src1RowStart = src1.rowPointers[i];
            int src1RowEnd = src1.rowPointers[i + 1];
            int src2RowStart = src2.rowPointers[i];
            int src2RowEnd = src2.rowPointers[i + 1];

            int src1Index = src1RowStart;
            int src2Index = src2RowStart;

            while (src1Index < src1RowEnd || src2Index < src2RowEnd) {
                // Skip explicit zeros in both matrices
                while (src1Index < src1RowEnd && src1.entries[src1Index] == 0) {
                    src1Index++;
                }
                while (src2Index < src2RowEnd && src2.entries[src2Index] == 0) {
                    src2Index++;
                }

                if (src1Index < src1RowEnd && src2Index < src2RowEnd) {
                    int src1Col = src1.colIndices[src1Index];
                    int src2Col = src2.colIndices[src2Index];

                    if (src1Col != src2Col) {
                        System.out.println("false return 2");
                        return false;
                    }

                    double src1Value = src1.entries[src1Index];
                    double src2Value = src2.entries[src2Index];

                    if (Double.compare(src1Value, src2Value) != 0) {
                        return false;
                    }

                    src1Index++;
                    src2Index++;
                } else if (src1Index < src1RowEnd) {
                    // Remaining entries in src1 row
                    if (src1.entries[src1Index] != 0) {
                        System.out.println("false return 3");
                        return false;
                    }
                    src1Index++;
                } else if (src2Index < src2RowEnd) {
                    // Remaining entries in src2 row
                    if (src2.entries[src2Index] != 0) {
                        System.out.println("false return 4");
                        return false;
                    }
                    src2Index++;
                }
            }
        }
        return true;
    }
}
