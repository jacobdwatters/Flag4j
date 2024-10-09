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

package org.flag4j.linalg.operations.sparse;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.CsrFieldMatrixBase;
import org.flag4j.arrays.sparse.CsrFieldMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.util.ErrorMessages;

import java.util.*;

/**
 * Contains common utility functions for working with sparse matrices.
 */
public final class SparseUtils {

    public SparseUtils() {
        // Utility class cannot be instanced.
        throw new IllegalArgumentException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Creates a HashMap where the keys are row indices and the value is a list of all indices in src with that row
     * index.
     * @param nnz Number of non-zero entries in the sparse matrix.
     * @param rowIndices Row indices of sparse matrix.
     * @return A HashMap where the keys are row indices and the value is a list of all indices in {@code src} with that row
     * index.
     */
    public static Map<Integer, List<Integer>> createMap(int nnz, int[] rowIndices) {
        Map<Integer, List<Integer>> map = new HashMap<>();

        for(int j=0; j<nnz; j++) {
            int r2 = rowIndices[j]; // = k
            map.computeIfAbsent(r2, x -> new ArrayList<>()).add(j);
        }

        return map;
    }


    /**
     * Sorts the non-zero entries and column indices of a sparse CSR matrix lexicographically by row and column index. The row
     * pointers in the CSR matrix are assumed to be correct already.
     * @param entries Non-zero entries of the CSR matrix.
     * @param rowPointers Row pointer array of the CSR matrix. Stores the starting index for each row of the CSR matrix in {@code entries}
     * and
     * @param colIndices Non-zero column indices of the CSR matrix.
     */
    public static void sortCsrMatrix(double[] entries, int[] rowPointers, int[] colIndices) {
        for (int row = 0; row < rowPointers.length - 1; row++) {
            int start = rowPointers[row];
            int end = rowPointers[row + 1];

            // Create an array of indices for sorting
            Integer[] indices = new Integer[end - start];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = start + i;
            }

            // Sort the indices based on the corresponding colIndices entries
            Arrays.sort(indices, (i, j) -> Integer.compare(colIndices[i], colIndices[j]));

            // Reorder colIndices and entries based on sorted indices
            int[] sortedColIndex = new int[end - start];
            double[] sortedValues = new double[end - start];
            for (int i = 0; i < indices.length; i++) {
                sortedColIndex[i] = colIndices[indices[i]];
                sortedValues[i] = entries[indices[i]];
            }

            // Copy sorted arrays back to the original colIndices and entries
            System.arraycopy(sortedColIndex, 0, colIndices, start, sortedColIndex.length);
            System.arraycopy(sortedValues, 0, entries, start, sortedValues.length);
        }
    }


    /**
     * Sorts the non-zero entries and column indices of a sparse CSR matrix lexicographically by row and column index. The row
     * pointers in the CSR matrix are assumed to be correct already.
     * @param entries Non-zero entries of the CSR matrix.
     * @param rowPointers Row pointer array of the CSR matrix. Stores the starting index for each row of the CSR matrix in {@code entries}
     * and
     * @param colIndices Non-zero column indices of the CSR matrix.
     */
    public static void sortCsrMatrix(Object[] entries, int[] rowPointers, int[] colIndices) {
        for (int row = 0; row < rowPointers.length - 1; row++) {
            int start = rowPointers[row];
            int end = rowPointers[row + 1];

            // Create an array of indices for sorting
            Integer[] indices = new Integer[end - start];
            for (int i = 0; i < indices.length; i++)
                indices[i] = start + i;

            // Sort the indices based on the corresponding colIndices entries
            Arrays.sort(indices, (i, j) -> Integer.compare(colIndices[i], colIndices[j]));

            // Reorder colIndices and entries based on sorted indices
            int[] sortedColIndex = new int[end - start];
            Object[] sortedValues = new Object[end - start];
            for (int i = 0; i < indices.length; i++) {
                sortedColIndex[i] = colIndices[indices[i]];
                sortedValues[i] = entries[indices[i]];
            }

            // Copy sorted arrays back to the original colIndices and entries
            System.arraycopy(sortedColIndex, 0, colIndices, start, sortedColIndex.length);
            System.arraycopy(sortedValues, 0, entries, start, sortedValues.length);
        }
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
    public static boolean CSREquals(CsrMatrix src1, CsrMatrix src2) {
        if(!src1.shape.equals(src2.shape)) return false;

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
                while (src1Index < src1RowEnd && src1.entries[src1Index] == 0)
                    src1Index++;

                while (src2Index < src2RowEnd && src2.entries[src2Index] == 0)
                    src2Index++;

                if (src1Index < src1RowEnd && src2Index < src2RowEnd) {
                    int src1Col = src1.colIndices[src1Index];
                    int src2Col = src2.colIndices[src2Index];
                    double src1Value = src1.entries[src1Index];
                    double src2Value = src2.entries[src2Index];

                    if(src1Col != src2Col || Double.compare(src1Value, src2Value) != 0) {
                        return false;
                    }

                    src1Index++;
                    src2Index++;
                } else if (src1Index < src1RowEnd) {
                    // Remaining entries in src1 row
                    if (src1.entries[src1Index] != 0) return false;

                    src1Index++;
                } else if (src2Index < src2RowEnd) {
                    // Remaining entries in src2 row
                    if (src2.entries[src2Index] != 0) return false;

                    src2Index++;
                }
            }
        }

        return true;
    }


    /**
     * <p>Checks if two {@link CsrFieldMatrix CSR Field matrices} are equal considering the fact that one may explicitly store zeros at
     * some position that the other does not store.</p>
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
    public static <T extends Field<T>> boolean CSREquals(CsrFieldMatrixBase<? ,?, ?, ?, T> src1,
                                                         CsrFieldMatrixBase<? ,?, ?, ?, T> src2) {
        if(!src1.shape.equals(src2.shape)) return false;
        final Complex128 ZERO = Complex128.ZERO;

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
                while (src1Index < src1RowEnd && src1.entries[src1Index].equals(ZERO))
                    src1Index++;

                while (src2Index < src2RowEnd && src2.entries[src2Index].equals(ZERO))
                    src2Index++;

                if (src1Index < src1RowEnd && src2Index < src2RowEnd) {
                    int src1Col = src1.colIndices[src1Index];
                    int src2Col = src2.colIndices[src2Index];

                    if(src1Col != src2Col || !src1.entries[src1Index].equals(src2.entries[src2Index]))
                        return false;

                    src1Index++;
                    src2Index++;
                } else if (src1Index < src1RowEnd) {
                    // Remaining entries in src1 row
                    if (!src1.entries[src1Index].equals(ZERO)) return false;
                    src1Index++;
                } else if (src2Index < src2RowEnd) {
                    // Remaining entries in src2 row
                    if (!src2.entries[src2Index].equals(ZERO)) return false;
                    src2Index++;
                }
            }
        }

        return true;
    }
}
