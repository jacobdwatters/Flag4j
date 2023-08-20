package com.flag4j.operations.sparse.real;

import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.SparseMatrix;
import com.flag4j.SparseVector;
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
 * This class provides methods for getting and setting elements and slices from/to a real sparse matrix.
 */
public class RealSparseMatrixGetSet {

    private RealSparseMatrixGetSet() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Gets the specified element from a sparse matrix.
     * @param src Source matrix to get value from.
     * @param row Row index of the value to get from the sparse matrix.
     * @param col Column index of the value to get from the sparse matrix.
     * @return The value in the sparse matrix at the specified indices.
     */
    public static double matrixGet(SparseMatrix src, int row, int col) {
        int idx = RealSparseElementSearch.matrixBinarySearch(src, row, col);
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
        int idx = RealSparseElementSearch.matrixBinarySearch(src, row, col);
        double[] destEntries;
        int[] destRowIndices;
        int[] destColIndices;

        if(idx < 0) {
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


    /**
     * Sets a specified row of a real sparse matrix to the values of a dense array.
     * @param src Source matrix to set the row of.
     * @param rowIdx Index of the row to set.
     * @param row Dense array containing the entries of the row to set.
     * @return A copy of the {@code src} matrix with the specified row set to the dense {@code row} array.
     */
    public static SparseMatrix setRow(SparseMatrix src, int rowIdx, double[] row) {
        ParameterChecks.assertIndexInBounds(src.numRows, rowIdx);
        ParameterChecks.assertEquals(src.numCols, row.length);

        int[] startEnd = RealSparseElementSearch.matrixFindRowStartEnd(src, rowIdx);
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
            System.arraycopy(ArrayUtils.intRange(0, src.numCols), 0, destColIndices, -start-1, row.length);
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
            System.arraycopy(ArrayUtils.intRange(0, src.numCols), 0, destColIndices, start, row.length);
            System.arraycopy(
                    src.colIndices, end,
                    destColIndices, start + row.length, destEntries.length-(start + row.length)
            );
        }

        return new SparseMatrix(src.shape.copy(), destEntries, destRowIndices, destColIndices);
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
    public static SparseMatrix setCol(SparseMatrix src, int colIdx, double[] col) {
        ParameterChecks.assertIndexInBounds(src.numCols, colIdx);
        ParameterChecks.assertEquals(src.numRows, col.length);

        Integer[] colIndices = new Integer[col.length];
        Arrays.fill(colIndices, colIdx);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<Double> destEntries = DoubleStream.of(col).boxed().collect(Collectors.toList());
        List<Integer> destRowIndices = IntStream.of(
                ArrayUtils.intRange(0, col.length)
        ).boxed().collect(Collectors.toList());
        List<Integer> destColIndices = new ArrayList<>(Arrays.asList(colIndices));

        return addNotInCol(destEntries, destRowIndices, destColIndices, src, colIdx);
    }


    /**
     * Sets a column of a sparse matrix to the values in a sparse tensor.
     * @param src Source matrix to set column of.
     * @param colIdx Index of the column to set.
     * @param col New entries for the specified column.
     * @return A copy of the {@code src} matrix with the specified column set to the {@code col} sparse vector.
     * @throws IllegalArgumentException If the {@code src} matrix does not have the same number of rows as total entries
     * in the {@code col} vector.
     */
    public static SparseMatrix setCol(SparseMatrix src, int colIdx, SparseVector col) {
        ParameterChecks.assertIndexInBounds(src.numCols, colIdx);
        ParameterChecks.assertEquals(src.numRows, col.size);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<Double> destEntries = ArrayUtils.toArrayList(col.entries);
        List<Integer> destRowIndices = ArrayUtils.toArrayList(col.indices);
        List<Integer> destColIndices = ArrayUtils.toArrayList(ArrayUtils.filledArray(col.nonZeroEntries(), colIdx));

        return addNotInCol(destEntries, destRowIndices, destColIndices, src, colIdx);
    }


    /**
     * Adds values from a sparse matrix to specified lists if the value is not within a specified column.
     * @param destEntries List to add non-zero entries from sparse matrix to.
     * @param destRowIndices List to add non-zero row indices from sparse matrix to.
     * @param destColIndices List to add non-zero column indices from sparse matrix to.
     * @param src The sparse matrix to get non-zero values and indices from.
     * @param colIdx Specified column to not add entries to the lists from.
     * @return A sparse matrix made from the resulting {@code destEntries, destRowIndices} and {@code destColIndices}
     * lists.
     */
    private static SparseMatrix addNotInCol(List<Double> destEntries, List<Integer> destRowIndices,
                                            List<Integer> destColIndices, SparseMatrix src, int colIdx) {
        for(int i=0; i<src.entries.length; i++) {
            // Add all entries which are not in the specified column.
            if(src.colIndices[i]!=colIdx) {
                destEntries.add(src.entries[i]);
                destRowIndices.add(src.rowIndices[i]);
                destColIndices.add(src.colIndices[i]);
            }
        }

        SparseMatrix dest = new SparseMatrix(
                src.shape.copy(),
                destEntries,
                destRowIndices,
                destColIndices
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
    public static SparseMatrix setSlice(SparseMatrix src, SparseMatrix values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.numRows + row);
        ParameterChecks.assertLessEq(src.numCols, values.numCols + col);

        // Initialize lists to new values for the specified slice.
        List<Double> entries = DoubleStream.of(values.entries).boxed().collect(Collectors.toList());
        List<Integer> rowIndices = ArrayUtils.toArrayList(ArrayUtils.shift(row, values.rowIndices));
        List<Integer> colIndices = ArrayUtils.toArrayList(ArrayUtils.shift(col, values.colIndices));

        int[] rowRange = ArrayUtils.intRange(row, values.numRows + row);
        int[] colRange = ArrayUtils.intRange(col, values.numCols + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure entries are properly sorted.
        SparseMatrix mat = new SparseMatrix(src.shape.copy(), entries, rowIndices, colIndices);
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
    public static SparseMatrix setSlice(SparseMatrix src, double[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.length + row);
        ParameterChecks.assertLessEq(src.numCols, values[0].length + col);

        // Flatten values.
        double[] flatValues = Arrays.stream(values)
                .flatMapToDouble(Arrays::stream)
                .toArray();
        int[] sliceRows = ArrayUtils.intRange(row, values.length + row);
        int[] sliceCols = ArrayUtils.intRange(col, values[0].length + col);

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
    public static SparseMatrix setSlice(SparseMatrix src, Matrix values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        ParameterChecks.assertLessEq(src.numRows, values.numRows + row);
        ParameterChecks.assertLessEq(src.numCols, values.numCols + col);

        int[] sliceRows = ArrayUtils.intRange(row, values.numRows + row);
        int[] sliceCols = ArrayUtils.intRange(col, values.numCols + col);

        return setSlice(src, values.entries, sliceRows, sliceCols, row, col);
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
    public static SparseMatrix setSlice(SparseMatrix src, int[][] values, int row, int col) {
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
        int[] sliceRows = ArrayUtils.intRange(row, values.length + row);
        int[] sliceCols = ArrayUtils.intRange(col, values[0].length + col);

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
    public static SparseMatrix setSlice(SparseMatrix src, Double[][] values, int row, int col) {
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

        int[] sliceRows = ArrayUtils.intRange(row, values.length + row);
        int[] sliceCols = ArrayUtils.intRange(col, values[0].length + col);

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
    public static SparseMatrix setSlice(SparseMatrix src, Integer[][] values, int row, int col) {
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

        int[] sliceRows = ArrayUtils.intRange(row, values.length + row);
        int[] sliceCols = ArrayUtils.intRange(col, values[0].length + col);

        return setSlice(src, flatValues, sliceRows, sliceCols, row, col);
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
    private static SparseMatrix setSlice(SparseMatrix src, double[] values,
                                         int[] sliceRows, int[] sliceCols, int row, int col) {
        // Copy vales and row/column indices (with appropriate shifting) to destination lists.
        List<Double> entries = DoubleStream.of(values).boxed().collect(Collectors.toList());
        List<Integer> rowIndices = ArrayUtils.toArrayList(sliceRows);
        List<Integer> colIndices = ArrayUtils.toArrayList(sliceCols);

        int[] rowRange = ArrayUtils.intRange(row, sliceRows.length + row);
        int[] colRange = ArrayUtils.intRange(col, sliceCols.length + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure entries are properly sorted.
        SparseMatrix mat = new SparseMatrix(src.shape.copy(), entries, rowIndices, colIndices);
        mat.sparseSort();

        return mat;
    }


    /**
     * Gets a specified row from this sparse matrix.
     * @param src Source sparse matrix to extract row from.
     * @param rowIdx Index of the row to extract from the {@code src} matrix.
     * @return Returns the specified row from this sparse matrix.
     */
    public static SparseVector getRow(SparseMatrix src, int rowIdx) {
        ParameterChecks.assertLessEq(src.numRows-1, rowIdx);

        List<Double> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i=0; i<src.entries.length; i++) {
            if(src.rowIndices[i]==rowIdx) {
                entries.add(src.entries[i]);
                indices.add(src.colIndices[i]);
            }
        }

        return new SparseVector(src.numCols, entries, indices);
    }


    /**
     * Gets a specified row range from this sparse matrix.
     * @param src Source sparse matrix to extract row from.
     * @param rowIdx Index of the row to extract from the {@code src} matrix.
     * @param start Staring column index of the column to be extracted (inclusive).
     * @param end Ending column index of the column to be extracted (exclusive)
     * @return Returns the specified column range from this sparse matrix.
     */
    public static SparseVector getRow(SparseMatrix src, int rowIdx, int start, int end) {
        ParameterChecks.assertLessEq(end-1, start);

        List<Double> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i=0; i<src.entries.length; i++) {
            if(src.rowIndices[i]==rowIdx && src.colIndices[i] >= start && src.colIndices[i] < end) {
                entries.add(src.entries[i]);
                indices.add(src.colIndices[i]);
            }
        }

        return new SparseVector(src.numRows, entries, indices);
    }


    /**
     * Gets a specified column from this sparse matrix.
     * @param src Source sparse matrix to extract column from.
     * @param colIdx Index of the column to extract from the {@code src} matrix.
     * @return Returns the specified column from this sparse matrix.
     */
    public static SparseVector getCol(SparseMatrix src, int colIdx) {
        ParameterChecks.assertLessEq(src.numCols-1, colIdx);

        List<Double> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i=0; i<src.entries.length; i++) {
            if(src.colIndices[i]==colIdx) {
                entries.add(src.entries[i]);
                indices.add(src.rowIndices[i]);
            }
        }

        return new SparseVector(src.numRows, entries, indices);
    }


    /**
     * Gets a specified column range from this sparse matrix.
     * @param src Source sparse matrix to extract column from.
     * @param colIdx Index of the column to extract from the {@code src} matrix.
     * @param start Staring row index of the column to be extracted (inclusive).
     * @param end Ending row index of the column to be extracted (exclusive)
     * @return Returns the specified column range from this sparse matrix.
     */
    public static SparseVector getCol(SparseMatrix src, int colIdx, int start, int end) {
        ParameterChecks.assertLessEq(end-1, start);

        List<Double> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i=0; i<src.entries.length; i++) {
            if(src.colIndices[i]==colIdx && src.rowIndices[i] >= start && src.rowIndices[i] < end) {
                entries.add(src.entries[i]);
                indices.add(src.rowIndices[i]);
            }
        }

        return new SparseVector(src.numRows, entries, indices);
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
    public static SparseMatrix getSlice(SparseMatrix src, int rowStart, int rowEnd, int colStart, int colEnd) {
        ParameterChecks.assertInRange(rowStart, 0, rowEnd-1.0, "rowStart");
        ParameterChecks.assertInRange(rowEnd, rowStart+1.0, src.numRows, "rowEnd");
        ParameterChecks.assertInRange(colStart, 0, colEnd-1.0, "colStart");
        ParameterChecks.assertInRange(colEnd, colStart+1.0, src.numRows, "colEnd");

        Shape shape = new Shape(src.numRows - (rowEnd-rowStart), src.numCols-(colEnd-colStart));
        List<Double> entries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        int start = RealSparseElementSearch.matrixBinarySearch(src, rowStart, colStart);

        for(int i=start; i<src.entries.length; i++) {
            if(inSlice(src.rowIndices[i], src.colIndices[i], rowStart, rowEnd, colStart, colEnd)) {
                entries.add(src.entries[i]);
                rowIndices.add(src.rowIndices[i]-rowStart);
                colIndices.add(src.colIndices[i]-colStart);
            }
        }

        return new SparseMatrix(shape, entries, rowIndices, colIndices);
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
    private static void copyValuesNotInSlice(SparseMatrix src, List<Double> entries, List<Integer> rowIndices,
                                             List<Integer> colIndices, int[] rowRange, int[] colRange) {
        // Copy values not in slice.
        for(int i=0; i<src.entries.length; i++) {
            if( !(ArrayUtils.contains(rowRange, src.rowIndices[i])
                    || ArrayUtils.contains(colRange, src.colIndices[i])) ) {
                // Then the entry is not in the slice so add it.
                entries.add(src.entries[i]);
                rowIndices.add(src.rowIndices[i]);
                colIndices.add(src.colIndices[i]);
            }
        }
    }
}
