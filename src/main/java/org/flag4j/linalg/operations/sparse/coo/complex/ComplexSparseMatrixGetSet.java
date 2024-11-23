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

package org.flag4j.linalg.operations.sparse.coo.complex;


import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.operations.sparse.SparseElementSearch;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class contains methods for getting/setting elements and slices from/to a complex sparse matrix.
 */
public final class ComplexSparseMatrixGetSet {

    private ComplexSparseMatrixGetSet() {
        // Hide default constructor in utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Sets a specified row of a complex sparse matrix to the values of a dense array.
     * @param src Source matrix to set the row of.
     * @param rowIdx Index of the row to set.
     * @param row Dense array containing the entries of the row to set.
     * @return A copy of the {@code src} matrix with the specified row set to the dense {@code row} array.
     */
    public static CooCMatrix setRow(CooCMatrix src, int rowIdx, double[] row) {
        ValidateParameters.ensureIndexInBounds(src.numRows, rowIdx);
        ValidateParameters.ensureEquals(src.numCols, row.length);

        int[] startEnd = SparseElementSearch.matrixFindRowStartEnd(src.rowIndices, rowIdx);
        int start = startEnd[0];
        int end = startEnd[1];

        Complex128[] destEntries;
        int[] destRowIndices ;
        int[] destColIndices;

        if(start<0) {
            // No entries with row index found.
            destEntries = new Complex128[src.entries.length + row.length];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.entries, 0, destEntries, 0, -start-1);
            ArrayUtils.arraycopy(row, 0, destEntries, -start-1, row.length);
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
            destEntries = new Complex128[src.entries.length + row.length - (end-start)];
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.entries, 0, destEntries, 0, start);
            ArrayUtils.arraycopy(row, 0, destEntries, start, row.length);
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

        return new CooCMatrix(src.shape, destEntries, destRowIndices, destColIndices);
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
    public static CooCMatrix setCol(CooCMatrix src, int colIdx, double[] col) {
        ValidateParameters.ensureIndexInBounds(src.numCols, colIdx);
        ValidateParameters.ensureEquals(src.numRows, col.length);

        // Initialize destination arrays_old with the new column and the appropriate indices.
        List<Field<Complex128>> destEntries = new ArrayList<>(src.entries.length);
        List<Integer> destRowIndices = new ArrayList<>(src.entries.length);
        List<Integer> destColIndices = new ArrayList<>(src.entries.length);

        // Add all entries in old matrix that are NOT in the specified column.
        for(int i=0; i<src.entries.length; i++) {
            if(src.colIndices[i]!=colIdx) {
                destEntries.add(src.entries[i]);
                destRowIndices.add(src.rowIndices[i]);
                destColIndices.add(src.colIndices[i]);
            }
        }

        int[] colIndices = new int[col.length];
        Arrays.fill(colIndices, colIdx);

        Complex128[] destEntriesArr = ArrayUtils.spliceDouble(destEntries, col, 0);
        int[] destRowIndicesArr = ArrayUtils.splice(destRowIndices, ArrayUtils.intRange(0, col.length), 0);
        int[] destColIndicesArr = ArrayUtils.splice(destColIndices, colIndices, 0);

        CooCMatrix dest = new CooCMatrix(
                src.shape,
                destEntriesArr,
                destRowIndicesArr,
                destColIndicesArr
        );

        dest.sortIndices();

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
    public static CooCMatrix setSlice(CooCMatrix src, CooMatrix values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        setSliceParamCheck(src, values, row, col);

        // Initialize lists to new values for the specified slice.
        List<Field<Complex128>> entries = ArrayUtils.toComplexArrayList(values.entries);
        List<Integer> rowIndices = ArrayUtils.toArrayList(ArrayUtils.shift(row, values.rowIndices));
        List<Integer> colIndices = ArrayUtils.toArrayList(ArrayUtils.shift(col, values.colIndices));

        int[] rowRange = ArrayUtils.intRange(row, values.numRows + row);
        int[] colRange = ArrayUtils.intRange(col, values.numCols + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure entries are properly sorted.
        CooCMatrix mat = new CooCMatrix(src.shape, entries, rowIndices, colIndices);
        mat.sortIndices();

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
    public static CooCMatrix setSlice(CooCMatrix src, double[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        setSliceParamCheck(src, values.length, values[0].length, row, col);

        // Flatten values.
        double[] flatValues = ArrayUtils.flatten(values);
        int[] sliceRows = ArrayUtils.intRange(row, values.length + row, values[0].length);
        int[] sliceCols = ArrayUtils.repeat(values.length, ArrayUtils.intRange(col, values[0].length + col));

        return setSlice(src, flatValues, values.length, values[0].length, sliceRows, sliceCols, row, col);
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
    public static CooCMatrix setSlice(CooCMatrix src, Matrix values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        setSliceParamCheck(src, values, row, col);

        int[] sliceRows = ArrayUtils.intRange(row, values.numRows + row, values.numCols);
        int[] sliceCols = ArrayUtils.repeat(values.numRows, ArrayUtils.intRange(col, values.numCols + col));

        return setSlice(src, values.entries, values.numRows, values.numCols, sliceRows, sliceCols, row, col);
    }



    /**
     * Sets a slice of a sparse matrix to values given in a 1d dense array.
     * @param src Source sparse matrix to copy non-slice from.
     * @param values Dense value for slice.
     * @param numRows Number of rows in the matrix represented by {@code values}.
     * @param numCols Number of columns in the matrix represented by {@code values}.
     * @param sliceRows Row indices for slice.
     * @param sliceCols Column indices for slice.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the specified values.
     */
    private static CooCMatrix setSlice(CooCMatrix src, double[] values, int numRows, int numCols,
                                          int[] sliceRows, int[] sliceCols, int row, int col) {
        // Copy vales and row/column indices (with appropriate shifting) to destination lists.
        List<Field<Complex128>> entries = ArrayUtils.toComplexArrayList(values);
        List<Integer> rowIndices = ArrayUtils.toArrayList(sliceRows);
        List<Integer> colIndices = ArrayUtils.toArrayList(sliceCols);

        int[] rowRange = ArrayUtils.intRange(row, numRows + row);
        int[] colRange = ArrayUtils.intRange(col, numCols + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure entries are properly sorted.
        CooCMatrix mat = new CooCMatrix(src.shape, entries, rowIndices, colIndices);
        mat.sortIndices();

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
    public static CooCMatrix setSlice(CooCMatrix src, Double[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        setSliceParamCheck(src, values.length, values[0].length, row, col);

        // Flatten values.
        double[] flatValues = ArrayUtils.unboxFlatten(values);

        int[] sliceRows = ArrayUtils.intRange(row, values.length + row, values[0].length);
        int[] sliceCols = ArrayUtils.repeat(values.length, ArrayUtils.intRange(col, values[0].length + col));

        return setSlice(src, flatValues, values.length, values[0].length, sliceRows, sliceCols, row, col);
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
    public static CooCMatrix setSlice(CooCMatrix src, Integer[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        setSliceParamCheck(src, values.length, values[0].length, row, col);

        // Flatten values.
        double[] flatValues = new double[values.length*values[0].length];
        int pos = 0;
        for(Integer[] vRow : values) {
            for(Integer d : vRow)
                flatValues[pos++] = d;
        }

        int[] sliceRows = ArrayUtils.intRange(row, values.length + row, values[0].length);
        int[] sliceCols = ArrayUtils.repeat(values.length, ArrayUtils.intRange(col, values[0].length + col));

        return setSlice(src, flatValues, values.length, values[0].length, sliceRows, sliceCols, row, col);
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
    public static CooCMatrix setSlice(CooCMatrix src, int[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        setSliceParamCheck(src, values.length, values[0].length, row, col);

        // Flatten values.
        double[] flatValues = new double[values.length*values[0].length];
        int pos = 0;
        for(int[] vRow : values) {
            for(int d : vRow)
                flatValues[pos++] = d;
        }
        int[] sliceRows = ArrayUtils.intRange(row, values.length + row, values[0].length);
        int[] sliceCols = ArrayUtils.repeat(values.length, ArrayUtils.intRange(col, values[0].length + col));

        return setSlice(src, flatValues, values.length, values[0].length, sliceRows, sliceCols, row, col);
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
    private static void copyValuesNotInSlice(CooCMatrix src, List<Field<Complex128>> entries, List<Integer> rowIndices,
                                             List<Integer> colIndices, int[] rowRange, int[] colRange) {
        // Copy values not in slice.
        for(int i=0, size=src.entries.length; i<size; i++) {
            if( !(ArrayUtils.contains(rowRange, src.rowIndices[i])
                    && ArrayUtils.contains(colRange, src.colIndices[i])) ) {
                // Then the entry is not in the slice so add it.
                entries.add(src.entries[i]);
                rowIndices.add(src.rowIndices[i]);
                colIndices.add(src.colIndices[i]);
            }
        }
    }


    private static <T extends MatrixMixin<?, ?, ?, ?>, U extends MatrixMixin<?, ?, ?, ?>>
    void setSliceParamCheck(T src, U values, int row, int col) {
        ValidateParameters.ensureIndexInBounds(src.numRows(), row);
        ValidateParameters.ensureIndexInBounds(src.numCols(), col);
        ValidateParameters.ensureLessEq(src.numRows(), values.numRows() + row);
        ValidateParameters.ensureLessEq(src.numCols(), values.numCols() + col);
    }


    private static <T extends MatrixMixin<?, ?, ?, ?>>
    void setSliceParamCheck(T src, int valueRows, int valueCols, int row, int col) {
        ValidateParameters.ensureIndexInBounds(src.numRows(), row);
        ValidateParameters.ensureIndexInBounds(src.numCols(), col);
        ValidateParameters.ensureLessEq(src.numRows(), valueRows + row);
        ValidateParameters.ensureLessEq(src.numCols(), valueCols + col);
    }
}
