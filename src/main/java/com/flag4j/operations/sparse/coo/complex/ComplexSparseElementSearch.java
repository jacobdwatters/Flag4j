package com.flag4j.operations.sparse.coo.complex;

import com.flag4j.CooCMatrix;
import com.flag4j.util.ErrorMessages;

import java.util.Arrays;

public class ComplexSparseElementSearch {

    private ComplexSparseElementSearch() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
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
    public static int matrixBinarySearch(CooCMatrix src, int rowKey, int colKey) {
        int rowIdx = Arrays.binarySearch(src.rowIndices, rowKey);

        if(rowIdx<0) return rowIdx;

        // Find range of same valued row indices.
        int lowerBound = rowIdx;
        for(int i=rowIdx; i>=0; i--) {
            if(src.rowIndices[i] == rowKey) {
                lowerBound = i;
            } else {
                break;
            }
        }

        int upperBound = rowIdx + 1;
        for(int i=upperBound; i<src.rowIndices.length; i++) {
            if(src.rowIndices[i] == rowKey) {
                upperBound = i;
            } else {
                break;
            }
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
    public static int[] matrixFindRowStartEnd(CooCMatrix src, int rowKey) {
        int rowIdx = Arrays.binarySearch(src.rowIndices, rowKey);

        if(rowIdx < 0) return new int[]{rowIdx, rowIdx}; // Row not found.

        // Find first entry with the specified row key.
        int lowerBound = rowIdx;
        for(int i=rowIdx; i>=0; i--) {
            if(src.rowIndices[i] == rowKey) {
                lowerBound = i;
            } else {
                break;
            }
        }

        int upperBound = rowIdx + 1;
        for(int i=upperBound; i<src.rowIndices.length; i++) {
            if(src.rowIndices[i] == rowKey) {
                upperBound = i+1;
            } else {
                break;
            }
        }

        return new int[]{lowerBound, upperBound};
    }
}
