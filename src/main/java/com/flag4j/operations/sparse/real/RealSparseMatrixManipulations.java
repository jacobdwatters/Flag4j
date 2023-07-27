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
     * @param src Source matrix to remove row from.
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
     * @param rowIdxs Indices of rows to remove from the {@code src} matrix. This array is assumed to be sorted and
     *                contain unique indices (however, this is not checked or enforced).
     *                If it is not sorted, call {@link java.util.Arrays#sort(int[])}
     *                first otherwise the behavior of this method is not defined.
     * @return A copy of the {@code src} matrix with the specified rows removed.
     */
    public static SparseMatrix removeRows(SparseMatrix src, int... rowIdxs) {
        Shape shape = new Shape(src.numRows-rowIdxs.length, src.numCols);
        List<Double> entries = new ArrayList<>(src.entries.length);
        List<Integer> rowIndices = new ArrayList<>(src.entries.length);
        List<Integer> colIndices = new ArrayList<>(src.entries.length);

        for(int i=0; i<src.entries.length; i++) {
            int idx = Arrays.binarySearch(rowIdxs, src.rowIndices[i]);

            if(idx < 0) {
                // Then copy the entry over and apply proper shift to row index.
                entries.add(src.entries[i]);
                rowIndices.add(src.rowIndices[i] + (idx+1));
                colIndices.add(src.colIndices[i]);
            }
        }

        return new SparseMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Removes a specified column from a sparse matrix.
     * @param src Source matrix to remove column from.
     * @param colIdx Column to remove from the {@code src} matrix.
     * @return A sparse matrix which has one less column than the {@code src} matrix with the specified column removed.
     */
    public static SparseMatrix removeCol(SparseMatrix src, int colIdx) {
        Shape shape = new Shape(src.numRows, src.numCols-1);
        List<Double> entries = new ArrayList<>(src.entries.length);
        List<Integer> rowIndices = new ArrayList<>(src.entries.length);
        List<Integer> colIndices = new ArrayList<>(src.entries.length);

        for(int i=0; i<src.entries.length; i++) {
            if(src.colIndices[i] != colIdx) {
                // Then entry is not in the specified column, so remove it.
                entries.add(src.entries[i]);
                rowIndices.add(src.rowIndices[i]);

                if(src.colIndices[i] < colIdx) colIndices.add(src.colIndices[i]);
                else colIndices.add(src.colIndices[i]-1);
            }
        }

        return new SparseMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Removes a list of specified columns from a sparse matrix.
     * @param src Source matrix to remove columns from.
     * @param colIdxs Columns to remove from the {@code src} matrix.
     * @return A copy of the {@code src} sparse matrix with the specified columns removed.
     */
    public static SparseMatrix removeCols(SparseMatrix src, int... colIdxs) {
        Shape shape = new Shape(src.numRows, src.numCols-1);
        List<Double> entries = new ArrayList<>(src.entries.length);
        List<Integer> rowIndices = new ArrayList<>(src.entries.length);
        List<Integer> colIndices = new ArrayList<>(src.entries.length);

        for(int i=0; i<src.entries.length; i++) {
            int idx = Arrays.binarySearch(colIdxs, src.colIndices[i]);

            if(idx < 0) {
                // Then entry is not in the specified column, so copy it with the appropriate column index shift.
                entries.add(src.entries[i]);
                rowIndices.add(src.rowIndices[i]);
                colIndices.add(src.colIndices[i] + (idx+1));
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
    private static void copyRanges(SparseMatrix src, double[] entries,
                                   int[] rowIndices, int[] colIndices, int[] startEnd) {
        if(startEnd[0] >= 0) {
            System.arraycopy(src.entries, 0, entries, 0, startEnd[0]);
            System.arraycopy(src.entries, startEnd[1], entries, startEnd[0], entries.length - startEnd[0]);

            System.arraycopy(src.rowIndices, 0, rowIndices, 0, startEnd[0]);
            System.arraycopy(src.rowIndices, startEnd[1], rowIndices, startEnd[0], entries.length - startEnd[0]);
            ArrayUtils.shiftRange(-1, rowIndices, startEnd[0], rowIndices.length); // Apply shift to row indices.

            System.arraycopy(src.colIndices, 0, colIndices, 0, startEnd[0]);
            System.arraycopy(src.colIndices, startEnd[1], colIndices, startEnd[0], entries.length - startEnd[0]);
        } else {
            System.arraycopy(src.entries, 0, entries, 0, entries.length);
            System.arraycopy(src.rowIndices, 0, rowIndices, 0, rowIndices.length);
            System.arraycopy(src.colIndices, 0, colIndices, 0, colIndices.length);

            ArrayUtils.shiftRange(-1, rowIndices, -startEnd[0]-1, rowIndices.length);
        }
    }


    /**
     * Swaps two rows, in place, in a sparse matrix.
     * @param src The source sparse matrix to swap rows within.
     * @param rowIdx1 Index of the first row in the swap.
     * @param rowIdx2 Index of the second row in the swap.
     * @return A reference to the {@code src} sparse matrix.
     */
    public static SparseMatrix swapRows(SparseMatrix src, int rowIdx1, int rowIdx2) {
        for(int i=0; i<src.entries.length; i++) {
            // Swap row indices.
            if(src.rowIndices[i]==rowIdx1) src.rowIndices[i] = rowIdx2;
            if(src.rowIndices[i]==rowIdx2) src.rowIndices[i] = rowIdx1;
        }

        // Ensure indices remain sorted properly.
        src.sparseSort();

        return src;
    }


    /**
     * Swaps two columns, in place, in a sparse matrix.
     * @param src The source sparse matrix to swap columns within.
     * @param colIdx1 Index of the first row in the swap.
     * @param colIdx2 Index of the second row in the swap.
     * @return A reference to the {@code src} sparse matrix.
     */
    public static SparseMatrix swapCols(SparseMatrix src, int colIdx1, int colIdx2) {
        for(int i=0; i<src.entries.length; i++) {
            // Swap row indices.
            if(src.colIndices[i]==colIdx1) src.colIndices[i] = colIdx2;
            if(src.colIndices[i]==colIdx2) src.colIndices[i] = colIdx1;
        }

        // Ensure indices remain sorted properly.
        src.sparseSort();

        return src;
    }
}
