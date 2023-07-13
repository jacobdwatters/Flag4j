package com.flag4j.operations.sparse.real;

import java.util.Arrays;

public class RealSparseElementSearch {


    /**
     * Gets the specified element from a sparse matrix.
     * @param entries Non-zero entries of the sparse matrix.
     * @param rowIndices Non-zero value row indices for the sparse matrix.
     * @param colIndices Non-zero value column indices for the sparse matrix.
     * @param row Row index of the value to get from the sparse matrix.
     * @param col Column index of the value to get from the sparse matrix.
     * @return
     */
    public static double matrixGet(double[] entries, int[] rowIndices, int[] colIndices, int row, int col) {
        // Find position of row index within the row indices if it exits.
        int rowIdx = Arrays.binarySearch(rowIndices, row);

        // No non-zero element with this row index exists.
        if(rowIdx < 0) return 0;

        // Find range of same valued row indices.
        int lowerBound = rowIdx;
        for(int i=rowIdx; i>0; i--) {
            if(rowIndices[i] == row) {
                lowerBound = i;
            } else {
                break;
            }
        }

        int upperBound = rowIdx + 1;
        for(int i=upperBound; i<rowIndices.length; i++) {
            if(rowIndices[i] == row) {
                upperBound = i;
            } else {
                break;
            }
        }

        int colIdx = Arrays.binarySearch(Arrays.copyOfRange(colIndices, lowerBound, upperBound), col);

        if(colIdx < 0) return 0;

        return entries[colIdx + lowerBound];
    }
}
