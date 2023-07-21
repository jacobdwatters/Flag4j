package com.flag4j.operations.sparse.complex;


import com.flag4j.SparseCMatrix;
import com.flag4j.SparseMatrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * This class contains methods for getting/setting elements and slices from/to a complex sparse matrix.
 */
public class ComplexSparseMatrixGetSet {

    private ComplexSparseMatrixGetSet() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Gets the specified element from a sparse matrix.
     * @param src Source matrix to get value from.
     * @param row Row index of the value to get from the sparse matrix.
     * @param col Column index of the value to get from the sparse matrix.
     * @return The value in the sparse matrix at the specified indices.
     */
    public static CNumber matrixGet(SparseCMatrix src, int row, int col) {
        int idx = ComplexSparseElementSearch.matrixBinarySearch(src, row, col);
        return idx<0 ? new CNumber() : src.entries[idx];
    }


    /**
     * Sets the specified element from a sparse matrix.
     * @param src Sparse matrix to set value in.
     * @param row Row index of the value to set in the sparse matrix.
     * @param col Column index of the value to set in the sparse matrix.
     * @param value Value to set.
     * @return The
     */
    public static SparseCMatrix matrixSet(SparseCMatrix src, int row, int col, CNumber value) {
        // Find position of row index within the row indices if it exits.
        int idx = ComplexSparseElementSearch.matrixBinarySearch(src, row, col);
        CNumber[] destEntries;
        int[] destRowIndices;
        int[] destColIndices;

        if(idx < 0) {
            System.out.println(idx);

            // No non-zero element with these indices exists. Insert new value.
            destEntries = new CNumber[src.entries.length + 1];
            ArrayUtils.arraycopy(src.entries, 0, destEntries, 0, -idx-1);
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
            destEntries = ArrayUtils.copyOf(src.entries);
            destEntries[idx] = value;
            destRowIndices = src.rowIndices.clone();
            destColIndices = src.colIndices.clone();
        }

        return new SparseCMatrix(src.shape.copy(), destEntries, destRowIndices, destColIndices);
    }


    /**
     * Sets a specified row of a complex sparse matrix to the values of a dense array.
     * @param src Source matrix to set the row of.
     * @param rowIdx Index of the row to set.
     * @param row Dense array containing the entries of the row to set.
     * @return A copy of the {@code src} matrix with the specified row set to the dense {@code row} array.
     */
    public static SparseCMatrix setRow(SparseCMatrix src, int rowIdx, double[] row) {
        ParameterChecks.assertIndexInBounds(src.numRows, rowIdx);
        ParameterChecks.assertEquals(src.numCols, row.length);

        int[] startEnd = ComplexSparseElementSearch.matrixFindRowStartEnd(src, rowIdx);
        int start = startEnd[0];
        int end = startEnd[1];

        double[] destEntries;
        int[] destRowIndices ;
        int[] destColIndices;

        if(start<0) {
            // No entries with row index found.
            destEntries = new double[src.entries.length + row.length];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.entries, 0, destEntries, 0, -start-1);
            System.arraycopy(row, 0, destEntries, -start-1, row.length);
            System.arraycopy(
                    src.entries, -start-1,
                    destEntries, -start-1+row.length, destEntries.length-(row.length - start - 1)
            );

            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, -start-1);
            Arrays.fill(destRowIndices, -start-1, -start-1+row.length, rowIdx);
            System.arraycopy(
                    src.rowIndices, -start-1,
                    destRowIndices, -start-1+row.length, destRowIndices.length-(row.length - start - 1)
            );

            System.arraycopy(src.colIndices, 0, destColIndices, 0, -start-1);
            System.arraycopy(ArrayUtils.rangeInt(0, src.numCols), 0, destColIndices, -start-1, row.length);
            System.arraycopy(
                    src.colIndices, -start-1,
                    destColIndices, -start-1+row.length, destColIndices.length-(row.length - start - 1)
            );

        } else {
            // Entries with row index found.
            destEntries = new double[src.entries.length + row.length - (end-start)];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.entries, 0, destEntries, 0, start);
            System.arraycopy(row, 0, destEntries, start, row.length);
            System.arraycopy(
                    src.entries, end,
                    destEntries, start + row.length, destEntries.length-(start + row.length)
            );

            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, start);
            Arrays.fill(destRowIndices, start, start+row.length, rowIdx);
            System.arraycopy(
                    src.rowIndices, end,
                    destRowIndices, start + row.length, destEntries.length-(start + row.length)
            );

            System.arraycopy(src.colIndices, 0, destColIndices, 0, start);
            System.arraycopy(ArrayUtils.rangeInt(0, src.numCols), 0, destColIndices, start, row.length);
            System.arraycopy(
                    src.colIndices, end,
                    destColIndices, start + row.length, destEntries.length-(start + row.length)
            );
        }

        return new SparseCMatrix(src.shape.copy(), destEntries, destRowIndices, destColIndices);
    }


    /**
     * Sets a column of a sparse matrix to the entries of a dense array.
     * @param src Source matrix to set column of.
     * @param colIdx The index of the column to set within the {@code src} matrix.
     * @param col The dense array containing the new column entries for the {@code src} array.
     * @return A copy of the {@code src} matrix with the specified column set to the dense array.
     * @throws IllegalArgumentException If the {@code colIdx} is not within the range of the matrix.
     * @throws IllegalArgumentException If the {@code col} array does not have the same length as the number of
     * rows in {@code src} matrix.
     */
    public static SparseCMatrix setCol(SparseCMatrix src, int colIdx, CNumber[] col) {
        ParameterChecks.assertIndexInBounds(src.numCols, colIdx);
        ParameterChecks.assertEquals(src.numRows, col.length);

        Integer[] colIndices = new Integer[col.length];
        Arrays.fill(colIndices, colIdx);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<CNumber> destEntries = new ArrayList<>(Arrays.asList(col));
        List<Integer> destRowIndices = IntStream.of(
                ArrayUtils.rangeInt(0, col.length)
        ).boxed().collect(Collectors.toList());
        List<Integer> destColIndices = new ArrayList<>(Arrays.asList(colIndices));

        // Add all entries in old matrix that are NOT in the specified column.
        for(int i=0; i<src.entries.length; i++) {
            if(src.colIndices[i]!=colIdx) {
                destEntries.add(src.entries[i]);
                destRowIndices.add(src.rowIndices[i]);
                destColIndices.add(src.colIndices[i]);
            }
        }

        SparseCMatrix dest = new SparseCMatrix(
                src.shape.copy(),
                destEntries.toArray(CNumber[]::new),
                destRowIndices.stream().mapToInt(Integer::intValue).toArray(),
                destColIndices.stream().mapToInt(Integer::intValue).toArray()
        );

        dest.sparseSort();

        return dest;
    }


    /**
     * Sets a column of a sparse matrix to the entries of a dense array.
     * @param src Source matrix to set column of.
     * @param colIdx The index of the column to set within the {@code src} matrix.
     * @param col The dense array containing the new column entries for the {@code src} array.
     * @return A copy of the {@code src} matrix with the specified column set to the dense array.
     * @throws IllegalArgumentException If the {@code colIdx} is not within the range of the matrix.
     * @throws IllegalArgumentException If the {@code col} array does not have the same length as the number of
     * rows in {@code src} matrix.
     */
    public static SparseCMatrix setCol(SparseCMatrix src, int colIdx, double[] col) {
        ParameterChecks.assertIndexInBounds(src.numCols, colIdx);
        ParameterChecks.assertEquals(src.numRows, col.length);

        Integer[] colIndices = new Integer[col.length];
        Arrays.fill(colIndices, colIdx);
        CNumber[] entries = new CNumber[col.length];
        ArrayUtils.copy2CNumber(col, entries);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<CNumber> destEntries = Arrays.asList(entries);
        List<Integer> destRowIndices = IntStream.of(
                ArrayUtils.rangeInt(0, col.length)
        ).boxed().collect(Collectors.toList());
        List<Integer> destColIndices = new ArrayList<>(Arrays.asList(colIndices));

        // Add all entries in old matrix that are NOT in the specified column.
        for(int i=0; i<src.entries.length; i++) {
            if(src.colIndices[i]!=colIdx) {
                destEntries.add(src.entries[i]);
                destRowIndices.add(src.rowIndices[i]);
                destColIndices.add(src.colIndices[i]);
            }
        }

        SparseCMatrix dest = new SparseCMatrix(
                src.shape.copy(),
                destEntries.toArray(CNumber[]::new),
                destRowIndices.stream().mapToInt(Integer::intValue).toArray(),
                destColIndices.stream().mapToInt(Integer::intValue).toArray()
        );

        dest.sparseSort();

        return dest;
    }
}
