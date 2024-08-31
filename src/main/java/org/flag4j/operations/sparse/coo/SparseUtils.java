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

package org.flag4j.operations.sparse.coo;

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
}
