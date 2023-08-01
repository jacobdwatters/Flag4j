package com.flag4j.operations.sparse.complex;


import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
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

        CNumber[] destEntries;
        int[] destRowIndices ;
        int[] destColIndices;

        if(start<0) {
            // No entries with row index found.
            destEntries = new CNumber[src.entries.length + row.length];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            ArrayUtils.arraycopy(src.entries, 0, destEntries, 0, -start-1);
            ArrayUtils.arraycopy(row, 0, destEntries, -start-1, row.length);
            ArrayUtils.arraycopy(
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
            destEntries = new CNumber[src.entries.length + row.length - (end-start)];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            ArrayUtils.arraycopy(src.entries, 0, destEntries, 0, start);
            ArrayUtils.arraycopy(row, 0, destEntries, start, row.length);
            ArrayUtils.arraycopy(
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
     * Sets a specified row of a complex sparse matrix to the values of a dense array.
     * @param src Source matrix to set the row of.
     * @param rowIdx Index of the row to set.
     * @param row Dense array containing the entries of the row to set.
     * @return A copy of the {@code src} matrix with the specified row set to the dense {@code row} array.
     */
    public static SparseCMatrix setRow(SparseCMatrix src, int rowIdx, CNumber[] row) {
        ParameterChecks.assertIndexInBounds(src.numRows, rowIdx);
        ParameterChecks.assertEquals(src.numCols, row.length);

        int[] startEnd = ComplexSparseElementSearch.matrixFindRowStartEnd(src, rowIdx);
        int start = startEnd[0];
        int end = startEnd[1];

        CNumber[] destEntries;
        int[] destRowIndices ;
        int[] destColIndices;

        if(start<0) {
            // No entries with row index found.
            destEntries = new CNumber[src.entries.length + row.length];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            ArrayUtils.arraycopy(src.entries, 0, destEntries, 0, -start-1);
            ArrayUtils.arraycopy(row, 0, destEntries, -start-1, row.length);
            ArrayUtils.arraycopy(
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
            destEntries = new CNumber[src.entries.length + row.length - (end-start)];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            ArrayUtils.arraycopy(src.entries, 0, destEntries, 0, start);
            ArrayUtils.arraycopy(row, 0, destEntries, start, row.length);
            ArrayUtils.arraycopy(
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
     * Sets a specified row of a complex sparse matrix to the values of a sparse vector.
     * @param src Source matrix to set the row of.
     * @param rowIdx Index of the row to set.
     * @param row Dense array containing the entries of the row to set.
     * @return A copy of the {@code src} matrix with the specified row set to the dense {@code row} array.
     */
    public static SparseCMatrix setRow(SparseCMatrix src, int rowIdx, SparseCVector row) {
        ParameterChecks.assertIndexInBounds(src.numRows, rowIdx);
        ParameterChecks.assertEquals(src.numCols, row.size);

        int[] startEnd = ComplexSparseElementSearch.matrixFindRowStartEnd(src, rowIdx);
        int start = startEnd[0];
        int end = startEnd[1];

        CNumber[] destEntries;
        int[] destRowIndices ;
        int[] destColIndices;

        if(start<0) {
            // No entries with row index found.
            destEntries = new CNumber[src.entries.length + row.entries.length];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            ArrayUtils.arraycopy(src.entries, 0, destEntries, 0, -start-1);
            ArrayUtils.arraycopy(row.entries, 0, destEntries, -start-1, row.entries.length);
            ArrayUtils.arraycopy(
                    src.entries, -start-1,
                    destEntries, -start-1+row.entries.length, destEntries.length-(row.entries.length - start - 1)
            );

            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, -start-1);
            Arrays.fill(destRowIndices, -start-1, -start-1+row.entries.length, rowIdx);
            System.arraycopy(
                    src.rowIndices, -start-1,
                    destRowIndices, -start-1+row.entries.length, destRowIndices.length-(row.entries.length - start - 1)
            );

            System.arraycopy(src.colIndices, 0, destColIndices, 0, -start-1);
            System.arraycopy(ArrayUtils.rangeInt(0, src.numCols), 0, destColIndices, -start-1, row.entries.length);
            System.arraycopy(
                    src.colIndices, -start-1,
                    destColIndices, -start-1+row.entries.length, destColIndices.length-(row.entries.length - start - 1)
            );

        } else {
            // Entries with row index found.
            destEntries = new CNumber[src.entries.length + row.entries.length - (end-start)];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            ArrayUtils.arraycopy(src.entries, 0, destEntries, 0, start);
            ArrayUtils.arraycopy(row.entries, 0, destEntries, start, row.entries.length);
            int length = destEntries.length - (start + row.entries.length);

            ArrayUtils.arraycopy(
                    src.entries, end,
                    destEntries, start + row.entries.length, length
            );

            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, start);
            Arrays.fill(destRowIndices, start, start+row.entries.length, rowIdx);
            System.arraycopy(
                    src.rowIndices, end,
                    destRowIndices, start + row.entries.length, length
            );

            System.arraycopy(src.colIndices, 0, destColIndices, 0, start);
            System.arraycopy(row.indices, 0, destColIndices, start, row.entries.length);
            System.arraycopy(
                    src.colIndices, end,
                    destColIndices, start + row.entries.length, length
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
                destEntries.add(src.entries[i].copy());
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
     * Sets a column of a sparse matrix to the entries of a sparse vector.
     * @param src Source matrix to set column of.
     * @param colIdx The index of the column to set within the {@code src} matrix.
     * @param col The dense array containing the new column entries for the {@code src} array.
     * @return A copy of the {@code src} matrix with the specified column set to the dense array.
     * @throws IllegalArgumentException If the {@code colIdx} is not within the range of the matrix.
     * @throws IllegalArgumentException If the {@code col} array does not have the same length as the number of
     * rows in {@code src} matrix.
     */
    public static SparseCMatrix setCol(SparseCMatrix src, int colIdx, SparseCVector col) {
        ParameterChecks.assertIndexInBounds(src.numCols, colIdx);
        ParameterChecks.assertEquals(src.numRows, col.size);

        Integer[] colIndices = new Integer[col.entries.length];
        Arrays.fill(colIndices, colIdx);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<CNumber> destEntries = ArrayUtils.toArrayList(col.entries);
        List<Integer> destRowIndices = ArrayUtils.toArrayList(col.indices);
        List<Integer> destColIndices = Arrays.asList(colIndices);

        // Add all entries in old matrix that are NOT in the specified column.
        for(int i=0; i<src.entries.length; i++) {
            if(src.colIndices[i]!=colIdx) {
                destEntries.add(src.entries[i].copy());
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
                destEntries.add(src.entries[i].copy());
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
     * Copies a sparse matrix and sets a slice of the sparse matrix to the entries of another sparse matrix.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} matrix.
     * @throws IllegalArgumentException If the {@code values} matrix does not fit in the {@code src}
     * matrix given the row and
     * column index.
     */
    public static SparseCMatrix setSlice(SparseCMatrix src, SparseMatrix values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.numRows + row);
        ParameterChecks.assertLessEq(src.numCols, values.numCols + col);

        // Initialize lists to new values for the specified slice.
        List<CNumber> entries = ArrayUtils.toComplexArrayList(values.entries);
        List<Integer> rowIndices = ArrayUtils.toArrayList(ArrayUtils.shift(row, values.rowIndices));
        List<Integer> colIndices = ArrayUtils.toArrayList(ArrayUtils.shift(col, values.colIndices));

        int[] rowRange = ArrayUtils.rangeInt(row, values.numRows + row);
        int[] colRange = ArrayUtils.rangeInt(col, values.numCols + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure entries are properly sorted.
        SparseCMatrix mat = new SparseCMatrix(src.shape.copy(), entries, rowIndices, colIndices);
        mat.sparseSort();

        return mat;
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the entries of another sparse matrix.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} matrix.
     * @throws IllegalArgumentException If the {@code values} matrix does not fit in the {@code src}
     * matrix given the row and
     * column index.
     */
    public static SparseCMatrix setSlice(SparseCMatrix src, SparseCMatrix values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.numRows + row);
        ParameterChecks.assertLessEq(src.numCols, values.numCols + col);

        // Initialize lists to new values for the specified slice.
        List<CNumber> entries = ArrayUtils.toArrayList(values.entries);
        List<Integer> rowIndices = ArrayUtils.toArrayList(ArrayUtils.shift(row, values.rowIndices));
        List<Integer> colIndices = ArrayUtils.toArrayList(ArrayUtils.shift(col, values.colIndices));

        int[] rowRange = ArrayUtils.rangeInt(row, values.numRows + row);
        int[] colRange = ArrayUtils.rangeInt(col, values.numCols + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure entries are properly sorted.
        SparseCMatrix mat = new SparseCMatrix(src.shape.copy(), entries, rowIndices, colIndices);
        mat.sparseSort();

        return mat;
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the entries of a dense array.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static SparseCMatrix setSlice(SparseCMatrix src, double[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.length + row);
        ParameterChecks.assertLessEq(src.numCols, values[0].length + col);

        // Flatten values.
        double[] flatValues = Arrays.stream(values)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        int[] sliceRows = ArrayUtils.rangeInt(row, values.length + row);
        int[] sliceCols = ArrayUtils.rangeInt(col, values[0].length + col);

        return setSlice(src, flatValues, sliceRows, sliceCols, row, col);
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the entries of a dense array.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static SparseCMatrix setSlice(SparseCMatrix src, CNumber[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.length + row);
        ParameterChecks.assertLessEq(src.numCols, values[0].length + col);

        // Flatten values.
        CNumber[] flatValues = ArrayUtils.flatten(values);
        int[] sliceRows = ArrayUtils.rangeInt(row, values.length + row);
        int[] sliceCols = ArrayUtils.rangeInt(col, values[0].length + col);

        return setSlice(src, flatValues, sliceRows, sliceCols, row, col);
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the entries of a dense matrix.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense matrix containing values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static SparseCMatrix setSlice(SparseCMatrix src, Matrix values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.numRows + row);
        ParameterChecks.assertLessEq(src.numCols, values.numCols + col);

        int[] sliceRows = ArrayUtils.rangeInt(row, values.numRows + row);
        int[] sliceCols = ArrayUtils.rangeInt(col, values.numCols + col);

        return setSlice(src, values.entries, sliceRows, sliceCols, row, col);
    }



    /**
     * Sets a slice of a sparse matrix to values given in a 1d dense array.
     * @param src Source sparse matrix to copy non-slice from.
     * @param values Dense value for slice.
     * @param sliceRows Row indices for slice.
     * @param sliceCols Column indices for slice.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the specified values.
     */
    private static SparseCMatrix setSlice(SparseCMatrix src, double[] values,
                                         int[] sliceRows, int[] sliceCols, int row, int col) {
        // Copy vales and row/column indices (with appropriate shifting) to destination lists.
        List<CNumber> entries = ArrayUtils.toComplexArrayList(values);
        List<Integer> rowIndices = IntStream.of(sliceRows).boxed().collect(Collectors.toList());
        List<Integer> colIndices = IntStream.of(sliceCols).boxed().collect(Collectors.toList());

        int[] rowRange = ArrayUtils.rangeInt(row, sliceRows.length + row);
        int[] colRange = ArrayUtils.rangeInt(col, sliceCols.length + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure entries are properly sorted.
        SparseCMatrix mat = new SparseCMatrix(src.shape.copy(), entries, rowIndices, colIndices);
        mat.sparseSort();

        return mat;
    }


    /**
     * Sets a slice of a sparse matrix to values given in a 1d dense array.
     * @param src Source sparse matrix to copy non-slice from.
     * @param values Dense value for slice.
     * @param sliceRows Row indices for slice.
     * @param sliceCols Column indices for slice.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the specified values.
     */
    private static SparseCMatrix setSlice(SparseCMatrix src, CNumber[] values,
                                          int[] sliceRows, int[] sliceCols, int row, int col) {
        // Copy vales and row/column indices (with appropriate shifting) to destination lists.
        List<CNumber> entries = ArrayUtils.toArrayList(values);
        List<Integer> rowIndices = IntStream.of(sliceRows).boxed().collect(Collectors.toList());
        List<Integer> colIndices = IntStream.of(sliceCols).boxed().collect(Collectors.toList());

        int[] rowRange = ArrayUtils.rangeInt(row, sliceRows.length + row);
        int[] colRange = ArrayUtils.rangeInt(col, sliceCols.length + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure entries are properly sorted.
        SparseCMatrix mat = new SparseCMatrix(src.shape.copy(), entries, rowIndices, colIndices);
        mat.sparseSort();

        return mat;
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the entries of a dense array.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static SparseCMatrix setSlice(SparseCMatrix src, Double[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.length + row);
        ParameterChecks.assertLessEq(src.numCols, values[0].length + col);

        // Flatten values.
        double[] flatValues = new double[values.length*values[0].length];
        int pos = 0;
        for(Double[] vRow : values) {
            for(Double d : vRow) {
                flatValues[pos++] = d;
            }
        }

        int[] sliceRows = ArrayUtils.rangeInt(row, values.length + row);
        int[] sliceCols = ArrayUtils.rangeInt(col, values[0].length + col);

        return setSlice(src, flatValues, sliceRows, sliceCols, row, col);
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the entries of a dense array.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static SparseCMatrix setSlice(SparseCMatrix src, Integer[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.length + row);
        ParameterChecks.assertLessEq(src.numCols, values[0].length + col);

        // Flatten values.
        double[] flatValues = new double[values.length*values[0].length];
        int pos = 0;
        for(Integer[] vRow : values) {
            for(Integer d : vRow) {
                flatValues[pos++] = d;
            }
        }

        int[] sliceRows = ArrayUtils.rangeInt(row, values.length + row);
        int[] sliceCols = ArrayUtils.rangeInt(col, values[0].length + col);

        return setSlice(src, flatValues, sliceRows, sliceCols, row, col);
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the entries of a dense array.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static SparseCMatrix setSlice(SparseCMatrix src, int[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.length + row);
        ParameterChecks.assertLessEq(src.numCols, values[0].length + col);

        // Flatten values.
        double[] flatValues = new double[values.length*values[0].length];
        int pos = 0;
        for(int[] vRow : values) {
            for(int d : vRow) {
                flatValues[pos++] = d;
            }
        }
        int[] sliceRows = ArrayUtils.rangeInt(row, values.length + row);
        int[] sliceCols = ArrayUtils.rangeInt(col, values[0].length + col);

        return setSlice(src, flatValues, sliceRows, sliceCols, row, col);
    }


    /**
     * Gets a specified row from this sparse matrix.
     * @param src Source sparse matrix to extract row from.
     * @param rowIdx Index of the row to extract from the {@code src} matrix.
     * @return Returns the specified row from this sparse matrix.
     */
    public static SparseCVector getRow(SparseCMatrix src, int rowIdx) {
        ParameterChecks.assertLessEq(src.numRows-1, rowIdx);

        List<CNumber> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i=0; i<src.entries.length; i++) {
            if(src.rowIndices[i]==rowIdx) {
                entries.add(src.entries[i].copy());
                indices.add(src.colIndices[i]);
            }
        }

        return new SparseCVector(src.numCols, entries, indices);
    }



    /**
     * Gets a specified row range from this sparse matrix.
     * @param src Source sparse matrix to extract row from.
     * @param rowIdx Index of the row to extract from the {@code src} matrix.
     * @param start Staring column index of the column to be extracted (inclusive).
     * @param end Ending column index of the column to be extracted (exclusive)
     * @return Returns the specified column range from this sparse matrix.
     */
    public static SparseCVector getRow(SparseCMatrix src, int rowIdx, int start, int end) {
        ParameterChecks.assertLessEq(end-1, start);

        List<CNumber> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i=0; i<src.entries.length; i++) {
            if(src.rowIndices[i]==rowIdx && src.colIndices[i] >= start && src.colIndices[i] < end) {
                entries.add(src.entries[i].copy());
                indices.add(src.colIndices[i]);
            }
        }

        return new SparseCVector(src.numRows, entries, indices);
    }


    /**
     * Gets a specified column from this sparse matrix.
     * @param src Source sparse matrix to extract column from.
     * @param colIdx Index of the column to extract from the {@code src} matrix.
     * @return Returns the specified column from this sparse matrix.
     */
    public static SparseCVector getCol(SparseCMatrix src, int colIdx) {
        ParameterChecks.assertLessEq(src.numCols-1, colIdx);

        List<CNumber> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i=0; i<src.entries.length; i++) {
            if(src.colIndices[i]==colIdx) {
                entries.add(src.entries[i].copy());
                indices.add(src.rowIndices[i]);
            }
        }

        return new SparseCVector(src.numRows, entries, indices);
    }


    /**
     * Gets a specified column range from this sparse matrix.
     * @param src Source sparse matrix to extract column from.
     * @param colIdx Index of the column to extract from the {@code src} matrix.
     * @param start Staring row index of the column to be extracted (inclusive).
     * @param end Ending row index of the column to be extracted (exclusive)
     * @return Returns the specified column range from this sparse matrix.
     */
    public static SparseCVector getCol(SparseCMatrix src, int colIdx, int start, int end) {
        ParameterChecks.assertLessEq(end-1, start);

        List<CNumber> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i=0; i<src.entries.length; i++) {
            if(src.colIndices[i]==colIdx && src.rowIndices[i] >= start && src.rowIndices[i] < end) {
                entries.add(src.entries[i].copy());
                indices.add(src.rowIndices[i]);
            }
        }

        return new SparseCVector(src.numRows, entries, indices);
    }


    /**
     * Gets a specified rectangular slice of a sparse matrix.
     * @param src Sparse matrix to extract slice from.
     * @param rowStart Starting row index of the slice (inclusive).
     * @param rowEnd Ending row index of the slice (exclusive).
     * @param colStart Staring column index of a slice (inclusive).
     * @param colEnd Ending column index of the slice (exclusive).
     * @return The specified slice of the sparse matrix.
     */
    public static SparseCMatrix getSlice(SparseCMatrix src, int rowStart, int rowEnd, int colStart, int colEnd) {
        ParameterChecks.assertInRange(rowStart, 0, rowEnd-1, "rowStart");
        ParameterChecks.assertInRange(rowEnd, rowStart+1, src.numRows, "rowEnd");
        ParameterChecks.assertInRange(colStart, 0, colEnd-1, "colStart");
        ParameterChecks.assertInRange(colEnd, colStart+1, src.numRows, "colEnd");

        Shape shape = new Shape(src.numRows - (rowEnd-rowStart), src.numCols-(colEnd-colStart));
        List<CNumber> entries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        int start = ComplexSparseElementSearch.matrixBinarySearch(src, rowStart, colStart);

        for(int i=start; i<src.entries.length; i++) {
            if(inSlice(src.rowIndices[i], src.colIndices[i], rowStart, rowEnd, colStart, colEnd)) {
                entries.add(src.entries[i].copy());
                rowIndices.add(src.rowIndices[i]-rowStart);
                colIndices.add(src.colIndices[i]-colStart);
            }
        }

        return new SparseCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Checks if an index is in the specified slice.
     * @param row Row index of value.
     * @param col Column index of value.
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending column index of slice (exclusive).
     * @return True if the indices are in the slice. False otherwise.
     */
    private static boolean inSlice(int row, int col, int rowStart, int rowEnd, int colStart, int colEnd) {
        return row >= rowStart && row < rowEnd && col >= colStart && col < colEnd;
    }


    /**
     * Copies values in sparse matrix which do not fall in the specified row and column ranges.
     * @param src Source sparse matrix to copy from.
     * @param entries Destination list to add copied values to.
     * @param rowIndices Destination list to add copied row indices to.
     * @param colIndices Destination list to add copied column indices to.
     * @param rowRange List of row indices to NOT copy from.
     * @param colRange List of column indices to NOT copy from.
     */
    private static void copyValuesNotInSlice(SparseCMatrix src, List<CNumber> entries, List<Integer> rowIndices,
                                             List<Integer> colIndices, int[] rowRange, int[] colRange) {
        // Copy values not in slice.
        for(int i=0; i<src.entries.length; i++) {
            if( !(ArrayUtils.contains(rowRange, src.rowIndices[i])
                    || ArrayUtils.contains(colRange, src.colIndices[i])) ) {
                // Then the entry is not in the slice so add it.
                entries.add(src.entries[i].copy());
                rowIndices.add(src.rowIndices[i]);
                colIndices.add(src.colIndices[i]);
            }
        }
    }
}
