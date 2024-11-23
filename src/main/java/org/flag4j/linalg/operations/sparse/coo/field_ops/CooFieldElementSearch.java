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

package org.flag4j.linalg.operations.sparse.coo.field_ops;

import org.flag4j.arrays.backend.field.AbstractCooFieldMatrix;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * Utility class for searching for specific elements within a sparse COO matrix.
 */
public final class CooFieldElementSearch {

    private CooFieldElementSearch() {
        // Hide default constructor in utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Preforms a binary search along the row and column indices of the non-zero values of a sparse matrix for the location
     * of an entry with the specified target indices.
     *
     * @param src Source matrix to search within.
     * @param rowKey Target row index.
     * @param colKey Target col index.
     * @return The location of the non-zero element (within the non-zero values array of {@code src}) with the specified
     *         row and column indices. If this value does not exist, then <code>(-(<i>insertion point</i>) - 1)</code>
     *         will be returned. The <i>insertion point</i> is defined as the point at which the
     *         value, with the row and column key, would be inserted into the array: the index of the first
     *         element greater than the key, or {@code src.entries.length} if all
     *         elements in the array are less than the specified key.  Note
     *         that this guarantees that the return value will be &gt;= 0 if
     *         and only if the key is found.
     */
    public static int matrixBinarySearch(AbstractCooFieldMatrix<?, ?, ?, ?> src, int rowKey, int colKey) {
        int rowIdx = Arrays.binarySearch(src.rowIndices, rowKey);
        if(rowIdx<0) return rowIdx;

        // Find range of same valued row indices.
        int lowerBound = rowIdx;
        for(int i=rowIdx; i>=0; i--) {
            if(src.rowIndices[i] == rowKey) lowerBound = i;
            else break;
        }

        int upperBound = rowIdx + 1;
        for(int i=upperBound; i<src.rowIndices.length; i++) {
            if(src.rowIndices[i] == rowKey) upperBound = i;
            else break;
        }

        int colIdx = Arrays.binarySearch(Arrays.copyOfRange(src.colIndices, lowerBound, upperBound), colKey);

        if(colIdx < 0) return colIdx-lowerBound;

        return colIdx + lowerBound;
    }


    /**
     * Finds the indices of the first and last non-zero element in the specified row of a sparse matrix. If there is no non-zero
     * element in the sparse matrix at the specified row, negative values will be returned.
     * @param src The source sparse matrix to search within.
     * @param rowKey Index of the row to search for within the row indices of the {@code src} matrix.
     * @return If it exists, the first and last index of the non-zero element in the sparse matrix which has the specified
     * {@code rowKey} as its row index.
     */
    public static int[] matrixFindRowStartEnd(AbstractCooFieldMatrix<?, ?, ?, ?> src, int rowKey) {
        int rowIdx = Arrays.binarySearch(src.rowIndices, rowKey);

        if(rowIdx < 0) return new int[]{rowIdx, rowIdx}; // Row not found.

        // Find first entry with the specified row key.
        int lowerBound = rowIdx;
        for(int i=rowIdx; i>=0; i--) {
            if(src.rowIndices[i] == rowKey) lowerBound = i;
            else break;
        }

        int upperBound = rowIdx + 1;
        for(int i=upperBound; i<src.rowIndices.length; i++) {
            if(src.rowIndices[i] == rowKey) upperBound = i+1;
            else break;
        }

        return new int[]{lowerBound, upperBound};
    }
}
