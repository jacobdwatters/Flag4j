package com.flag4j.operations.sparse.real;

import com.flag4j.SparseMatrix;

import java.util.Arrays;

public class RealSparseElementSearch {


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
    public static int matrixBinarySearch(SparseMatrix src, int rowKey, int colKey) {
        int rowIdx = Arrays.binarySearch(src.rowIndices, rowKey);

        if(rowIdx<0) return rowIdx;

        // Find range of same valued row indices.
        int lowerBound = rowIdx;
        for(int i=rowIdx; i>0; i--) {
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
     * Gets the specified element from a sparse matrix.
     * @param src Source matrix to get value from.
     * @param row Row index of the value to get from the sparse matrix.
     * @param col Column index of the value to get from the sparse matrix.
     * @return The value in the sparse matrix at the specified indices.
     */
    public static double matrixGet(SparseMatrix src, int row, int col) {
        int idx = matrixBinarySearch(src, row, col);
        return idx<0 ? 0 : src.entries[idx];
    }


    /**
     * Sets the specified element from a sparse matrix.
     * @param src Sparse matrix to set value in.
     * @param row Row index of the value to set in the sparse matrix.
     * @param col Column index of the value to set in the sparse matrix.
     * @param value Value to set.
     * @return The
     */
    public static SparseMatrix matrixSet(SparseMatrix src, int row, int col, double value) {
        // Find position of row index within the row indices if it exits.
        int idx = matrixBinarySearch(src, row, col);
        double[] destEntries;
        int[] destRowIndices;
        int[] destColIndices;

        if(idx < 0) {
            System.out.println(idx);

            // No non-zero element with these indices exists. Insert new value.
            destEntries = new double[src.entries.length + 1];
            System.arraycopy(src.entries, 0, destEntries, 0, -idx-1);
            destEntries[-idx-1] = value;
            System.arraycopy(src.entries, -idx-1, destEntries, -idx, src.entries.length+idx+1);

            destRowIndices = new int[src.entries.length + 1];
            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, -idx-1);
            destRowIndices[-idx-1] = row;
            System.arraycopy(src.rowIndices, -idx-1, destRowIndices, -idx, src.rowIndices.length+idx+1);

            destColIndices = new int[src.entries.length + 1];
            System.arraycopy(src.colIndices, 0, destColIndices, 0, -idx-1);
            destColIndices[-idx-1] = col;
            System.arraycopy(src.colIndices, -idx-1, destColIndices, -idx, src.colIndices.length+idx+1);
        } else {
            // Value with these indices exists. Simply update value.
            destEntries = Arrays.copyOf(src.entries, src.entries.length);
            destEntries[idx] = value;
            destRowIndices = src.rowIndices.clone();
            destColIndices = src.colIndices.clone();
        }

        return new SparseMatrix(src.shape.copy(), destEntries, destRowIndices, destColIndices);
    }
}
