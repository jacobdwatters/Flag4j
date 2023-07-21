package com.flag4j.operations.sparse.real;

import com.flag4j.Shape;
import com.flag4j.SparseMatrix;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.RandomTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class contains implementations for real sparse matrix manipulations.
 */
public class RealSparseMatrixManipulations {

    private RealSparseMatrixManipulations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Removes a specified row from a sparse matrix.
     * @param src Source matrix to remove row of.
     * @param rowIdx Row to remove from the {@code src} matrix.
     * @return A sparse matrix which has one less row than the {@code src} matrix with the specified row removed.
     */
    public static SparseMatrix removeRow(SparseMatrix src, int rowIdx) {
        Shape shape = new Shape(src.numRows-1, src.numCols);

        // Find the start and end index within the entries array which have the given row index.
        int[] startEnd = RealSparseElementSearch.matrixFindRowStartEnd(src, rowIdx);
        int size = src.entries.length - (startEnd[1]-startEnd[0]);

        // Initialize arrays.
        double[] entries = new double[size];
        int[] rowIndices = new int[size];
        int[] colIndices = new int[size];

        copyRanges(src, entries, rowIndices, colIndices, startEnd);

        return new SparseMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Removes multiple rows from a real sparse matrix.
     * @param src The source sparse matrix to remove rows from.
     * @param rowIdxs Indices of rows to remove from the {@code src} matrix.
     * @return
     */
    public static SparseMatrix removeRows(SparseMatrix src, int... rowIdxs) {
        Shape shape = new Shape(src.numRows-rowIdxs.length, src.numCols);
        List<Double> entries = new ArrayList<>(src.entries.length);
        List<Integer> rowIndices = new ArrayList<>(src.entries.length);
        List<Integer> colIndices = new ArrayList<>(src.entries.length);

        for(int i=0; i<src.entries.length; i++) {
            if(!ArrayUtils.contains(rowIdxs, src.rowIndices[i])) {
                // Then copy the entry over.
                entries.add(src.entries[i]);
                rowIndices.add(src.rowIndices[i]);
                colIndices.add(src.colIndices[i]);
            }
        }

        return new SparseMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * A helper method which copies from a sparse matrix to a set of three arrays (non-zero entries, row indices, and
     * column indices) but skips over a specified range.
     * @param src Source sparse matrix to copy from.
     * @param entries Array to copy {@code} src non-zero entries to.
     * @param rowIndices Array to copy {@code} src row indices entries to.
     * @param colIndices Array to copy {@code} src column indices entries to.
     * @param startEnd An array of length two specifying the {@code start} (inclusive) and {@code end} (exclusive)
     *                 indices of the range to skip during the copy.
     */
    private static void copyRanges(SparseMatrix src, double[] entries, int[]
            rowIndices, int[] colIndices, int[] startEnd) {

        if(startEnd[0] > 0) {
            System.arraycopy(src.entries, 0, entries, 0, startEnd[0]);
            System.arraycopy(src.entries, startEnd[1], entries, startEnd[0], entries.length - startEnd[0]);

            System.arraycopy(src.rowIndices, 0, rowIndices, 0, startEnd[0]);
            System.arraycopy(src.rowIndices, startEnd[1], rowIndices, startEnd[0], entries.length - startEnd[0]);

            System.arraycopy(src.colIndices, 0, colIndices, 0, startEnd[0]);
            System.arraycopy(src.colIndices, startEnd[1], colIndices, startEnd[0], entries.length - startEnd[0]);
        } else {
            System.arraycopy(src.entries, 0, entries, 0, entries.length);
            System.arraycopy(src.rowIndices, 0, rowIndices, 0, rowIndices.length);
            System.arraycopy(src.colIndices, 0, colIndices, 0, colIndices.length);
        }
    }
}
