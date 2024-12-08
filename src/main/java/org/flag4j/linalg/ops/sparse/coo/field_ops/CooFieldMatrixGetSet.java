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

package org.flag4j.linalg.ops.sparse.coo.field_ops;


import org.flag4j.algebraic_structures.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldVector;
import org.flag4j.linalg.ops.sparse.SparseElementSearch;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This class contains methods for getting/setting elements and slices from/to a sparse
 * {@link Field} matrix.
 */
public final class CooFieldMatrixGetSet {

    private CooFieldMatrixGetSet() {
        // Hide default constructor in utility class.

    }


    /**
     * Gets the specified element from a sparse matrix.
     * @param src Source matrix to get value from.
     * @param row Row index of the value to get from the sparse matrix.
     * @param col Column index of the value to get from the sparse matrix.
     * @return The value in the sparse matrix at the specified indices.
     */
    public static <V extends Field<V>> V matrixGet(AbstractCooFieldMatrix<?, ?, ?, V> src, int row, int col) {
        V zero = src.data.length > 0 ? src.data[0].getZero() : null;
        int idx = SparseElementSearch.matrixBinarySearch(src.rowIndices, src.colIndices, row, col);

        return idx<0 ? zero : src.data[idx];
    }


    /**
     * Sets the specified element from a sparse matrix.
     * @param src Sparse matrix to set value in.
     * @param row Row index of the value to set in the sparse matrix.
     * @param col Column index of the value to set in the sparse matrix.
     * @param value Value to set.
     * @return The
     */
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V>
    matrixSet(AbstractCooFieldMatrix<?, ?, ?, V> src, int row, int col, V value) {
        // Find position of row index within the row indices if it exits.
        int idx = SparseElementSearch.matrixBinarySearch(src.rowIndices, src.colIndices, row, col);
        V[] destEntries;
        int[] destRowIndices;
        int[] destColIndices;

        if(idx < 0) {
            idx = -idx - 1;

            // No non-zero element with these indices exists. Insert new value.
            destEntries = src.makeEmptyDataArray(src.data.length + 1);
            System.arraycopy(src.data, 0, destEntries, 0, idx);
            destEntries[idx] = value;
            System.arraycopy(src.data, idx, destEntries, -idx, src.data.length - idx);

            destRowIndices = new int[src.data.length + 1];
            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, idx);
            destRowIndices[idx] = row;
            System.arraycopy(src.rowIndices, idx, destRowIndices, -idx, src.rowIndices.length - idx);

            destColIndices = new int[src.data.length + 1];
            System.arraycopy(src.colIndices, 0, destColIndices, 0, idx);
            destColIndices[idx] = col;
            System.arraycopy(src.colIndices, idx, destColIndices, idx+1, src.colIndices.length - idx);
        } else {
            // Value with these indices exists. Simply update value.
            destEntries = Arrays.copyOf(src.data, src.data.length);
            destEntries[idx] = value;
            destRowIndices = src.rowIndices.clone();
            destColIndices = src.colIndices.clone();
        }

        return src.makeLikeTensor(src.shape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Sets a specified row of a complex sparse matrix to the values of a dense array.
     * @param src Source matrix to set the row of.
     * @param rowIdx Index of the row to set.
     * @param row Dense array containing the data of the row to set.
     * @return A copy of the {@code src} matrix with the specified row set to the dense {@code row} array.
     */
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V> setRow(
            AbstractCooFieldMatrix<?, ?, ?, V> src, int rowIdx, V[] row) {
        ValidateParameters.ensureIndicesInBounds(src.numRows, rowIdx);
        ValidateParameters.ensureEquals(src.numCols, row.length);

        int[] startEnd = SparseElementSearch.matrixFindRowStartEnd(src.rowIndices, rowIdx);
        int start = startEnd[0];
        int end = startEnd[1];

        V[] destEntries;
        int[] destRowIndices ;
        int[] destColIndices;

        if(start<0) {
            // No data with row index found.
            destEntries = src.makeEmptyDataArray(src.data.length + row.length);
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.data, 0, destEntries, 0, -start-1);
            System.arraycopy(row, 0, destEntries, -start-1, row.length);
            System.arraycopy(
                    src.data, -start-1,
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
            destEntries = src.makeEmptyDataArray(src.data.length + row.length - (end-start));
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.data, 0, destEntries, 0, start);
            System.arraycopy(row, 0, destEntries, start, row.length);
            System.arraycopy(
                    src.data, end,
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

        return src.makeLikeTensor(src.shape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Sets a specified row of a complex sparse matrix to the values of a sparse vector.
     * @param src Source matrix to set the row of.
     * @param rowIdx Index of the row to set.
     * @param row Dense array containing the data of the row to set.
     * @return A copy of the {@code src} matrix with the specified row set to the dense {@code row} array.
     */
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V> setRow(
            AbstractCooFieldMatrix<?, ?, ?, V> src, int rowIdx, AbstractCooFieldVector<?, ?, ?, ?, V> row) {
        ValidateParameters.ensureIndicesInBounds(src.numRows, rowIdx);
        ValidateParameters.ensureEquals(src.numCols, row.size);

        int[] startEnd = SparseElementSearch.matrixFindRowStartEnd(src.rowIndices, rowIdx);
        int start = startEnd[0];
        int end = startEnd[1];

        V[] destEntries;
        int[] destRowIndices ;
        int[] destColIndices;

        if(start<0) {
            // No data with row index found.
            destEntries = src.makeEmptyDataArray(src.data.length + row.data.length);
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.data, 0, destEntries, 0, -start-1);
            System.arraycopy(row.data, 0, destEntries, -start-1, row.data.length);
            System.arraycopy(
                    src.data, -start-1,
                    destEntries, -start-1+row.data.length, destEntries.length-(row.data.length - start - 1)
            );

            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, -start-1);
            Arrays.fill(destRowIndices, -start-1, -start-1+row.data.length, rowIdx);
            System.arraycopy(
                    src.rowIndices, -start-1,
                    destRowIndices, -start-1+row.data.length, destRowIndices.length-(row.data.length - start - 1)
            );

            System.arraycopy(src.colIndices, 0, destColIndices, 0, -start-1);
            System.arraycopy(row.indices, 0, destColIndices, -start-1, row.data.length);
            System.arraycopy(
                    src.colIndices, -start-1,
                    destColIndices, -start-1+row.data.length, destColIndices.length-(row.data.length - start - 1)
            );

        } else {
            // Entries with row index found.
            destEntries = src.makeEmptyDataArray(src.data.length + row.data.length - (end-start));
            destRowIndices = new int[destEntries.length];
            destColIndices = new int[destEntries.length];

            System.arraycopy(src.data, 0, destEntries, 0, start);
            System.arraycopy(row.data, 0, destEntries, start, row.data.length);
            int length = destEntries.length - (start + row.data.length);

            System.arraycopy(
                    src.data, end,
                    destEntries, start + row.data.length, length
            );

            System.arraycopy(src.rowIndices, 0, destRowIndices, 0, start);
            Arrays.fill(destRowIndices, start, start+row.data.length, rowIdx);
            System.arraycopy(
                    src.rowIndices, end,
                    destRowIndices, start + row.data.length, length
            );

            System.arraycopy(src.colIndices, 0, destColIndices, 0, start);
            System.arraycopy(row.indices, 0, destColIndices, start, row.data.length);
            System.arraycopy(
                    src.colIndices, end,
                    destColIndices, start + row.data.length, length
            );
        }

        return src.makeLikeTensor(src.shape, destEntries, destRowIndices, destColIndices);
    }


    /**
     * Sets a column of a sparse matrix to the data of a dense array.
     * @param src Source matrix to set column of.
     * @param colIdx The index of the column to set within the {@code src} matrix.
     * @param col The dense array containing the new column data for the {@code src} array.
     * @return A copy of the {@code src} matrix with the specified column set to the dense array.
     * @throws IllegalArgumentException If the {@code colIdx} is not within the range of the matrix.
     * @throws IllegalArgumentException If the {@code col} array does not have the same length as the number of
     * rows in {@code src} matrix.
     */
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V>
    setCol(AbstractCooFieldMatrix<?, ?, ?, V> src, int colIdx, V[] col) {
        ValidateParameters.ensureIndicesInBounds(src.numCols, colIdx);
        ValidateParameters.ensureEquals(src.numRows, col.length);

        Integer[] colIndices = new Integer[col.length];
        Arrays.fill(colIndices, colIdx);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<V> destEntries = Arrays.asList(col);
        List<Integer> destRowIndices = IntStream.of(
                ArrayUtils.intRange(0, col.length)
        ).boxed().collect(Collectors.toList());
        List<Integer> destColIndices = new ArrayList<>(Arrays.asList(colIndices));

        // Add all data in old matrix that are NOT in the specified column.
        for(int i = 0; i<src.data.length; i++) {
            if(src.colIndices[i]!=colIdx) {
                destEntries.add(src.data[i]);
                destRowIndices.add(src.rowIndices[i]);
                destColIndices.add(src.colIndices[i]);
            }
        }

        AbstractCooFieldMatrix<?, ?, ?, V> dest = src.makeLikeTensor(src.shape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure the indices are sorted properly.

        return dest;
    }


    /**
     * Sets a column of a sparse matrix to the data of a sparse vector.
     * @param src Source matrix to set column of.
     * @param colIdx The index of the column to set within the {@code src} matrix.
     * @param col The dense array containing the new column data for the {@code src} array.
     * @return A copy of the {@code src} matrix with the specified column set to the dense array.
     * @throws IllegalArgumentException If the {@code colIdx} is not within the range of the matrix.
     * @throws IllegalArgumentException If the {@code col} array does not have the same length as the number of
     * rows in {@code src} matrix.
     */
    public static <T extends Field<T>> AbstractCooFieldMatrix<?, ?, ?, T> setCol(
            AbstractCooFieldMatrix<?, ?, ?, T> src,
            int colIdx,
            AbstractCooFieldVector<?, ?, ?, ?, T> col) {
        ValidateParameters.ensureIndicesInBounds(src.numCols, colIdx);
        ValidateParameters.ensureEquals(src.numRows, col.size);

        int[] colIndices = new int[col.data.length];
        Arrays.fill(colIndices, colIdx);

        // Initialize destination arrays with the new column and the appropriate indices.
        List<T> destEntries = new ArrayList(Arrays.asList(col.data));
        List<Integer> destRowIndices = ArrayUtils.toArrayList(col.indices);
        List<Integer> destColIndices = ArrayUtils.toArrayList(colIndices);

        // Add all data in old matrix that are NOT in the specified column.
        for(int i = 0; i<src.data.length; i++) {
            if(src.colIndices[i]!=colIdx) {
                destEntries.add(src.data[i]);
                destRowIndices.add(src.rowIndices[i]);
                destColIndices.add(src.colIndices[i]);
            }
        }

        AbstractCooFieldMatrix<?, ?, ?, T> dest = src.makeLikeTensor(
                src.shape,
                destEntries,
                destRowIndices,
                destColIndices
        );
        dest.sortIndices();

        return dest;
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the data of another sparse matrix.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} matrix.
     * @throws IllegalArgumentException If the {@code values} matrix does not fit in the {@code src}
     * matrix given the row and
     * column index.
     */
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V> setSlice(
            AbstractCooFieldMatrix<?, ?, ?, V> src,
            AbstractCooFieldMatrix<?, ?, ?, V> values,
            int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        setSliceParamCheck(src, values, row, col);

        // Initialize lists to new values for the specified slice.
        List<V> entries = new ArrayList<>(Arrays.asList(values.data));
        List<Integer> rowIndices = ArrayUtils.toArrayList(ArrayUtils.shift(row, values.rowIndices));
        List<Integer> colIndices = ArrayUtils.toArrayList(ArrayUtils.shift(col, values.colIndices));

        int[] rowRange = ArrayUtils.intRange(row, values.numRows + row);
        int[] colRange = ArrayUtils.intRange(col, values.numCols + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure data are properly sorted.
        AbstractCooFieldMatrix<?, ?, ?, V> mat = src.makeLikeTensor(src.shape, entries, rowIndices, colIndices);
        mat.sortIndices();

        return mat;
    }


    /**
     * Copies a sparse matrix and sets a slice of the sparse matrix to the data of a dense array.
     * @param src Source sparse matrix to copy and set values of.
     * @param values Dense values of the slice to be set.
     * @param row Starting row index of slice.
     * @param col Starting column index of slice.
     * @return A copy of the {@code src} matrix with the specified slice set to the {@code values} array.
     * @throws IllegalArgumentException If the {@code values} array does not fit in the {@code src} matrix
     * given the row and column index.
     */
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V> setSlice(
            AbstractCooFieldMatrix<?, ?, ?, V> src, V[][] values, int row, int col) {
        // Ensure the values matrix fits inside the src matrix.
        setSliceParamCheck(src, values.length, values[0].length, row, col);

        // Flatten values.
        V[] flatValues = ArrayUtils.flatten(values);
        int[] sliceRows = ArrayUtils.intRange(row, values.length + row, values[0].length);
        int[] sliceCols = ArrayUtils.repeat(values.length, ArrayUtils.intRange(col, values[0].length + col));

        return setSlice(src, flatValues, values.length, values[0].length, sliceRows, sliceCols, row, col);
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
    private static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V> setSlice(
            AbstractCooFieldMatrix<?, ?, ?, V> src, V[] values,
            int numRows, int numCols, int[] sliceRows, int[] sliceCols, int row, int col) {
        // Copy vales and row/column indices (with appropriate shifting) to destination lists.
        List<V> entries = new ArrayList<>(Arrays.asList(values));
        List<Integer> rowIndices = ArrayUtils.toArrayList(sliceRows);
        List<Integer> colIndices = ArrayUtils.toArrayList(sliceCols);

        int[] rowRange = ArrayUtils.intRange(row, numRows + row);
        int[] colRange = ArrayUtils.intRange(col, numCols + col);

        copyValuesNotInSlice(src, entries, rowIndices, colIndices, rowRange, colRange);

        // Create matrix and ensure data are properly sorted.
        AbstractCooFieldMatrix<?, ?, ?, V> mat = src.makeLikeTensor(src.shape, entries, rowIndices, colIndices);
        mat.sortIndices();

        return mat;
    }


    /**
     * Gets a specified row from this sparse matrix.
     * @param src Source sparse matrix to extract row from.
     * @param rowIdx Index of the row to extract from the {@code src} matrix.
     * @return Returns the specified row from this sparse matrix.
     */
    public static <T extends Field<T>> AbstractCooFieldVector<?, ?, ?, ?, T> getRow(AbstractCooFieldMatrix<?, ?, ?, T> src, int rowIdx) {
        ValidateParameters.ensureIndicesInBounds(src.numRows, rowIdx);

        List<T> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i = 0; i<src.data.length; i++) {
            if(src.rowIndices[i]==rowIdx) {
                entries.add(src.data[i]);
                indices.add(src.colIndices[i]);
            }
        }

        return src.makeLikeVector(new Shape(src.numCols),
                entries.toArray(src.makeEmptyDataArray(entries.size())),
                ArrayUtils.fromIntegerList(indices));
    }


    /**
     * Gets a specified row range from this sparse matrix.
     * @param src Source sparse matrix to extract row from.
     * @param rowIdx Index of the row to extract from the {@code src} matrix.
     * @param start Staring column index of the column to be extracted (inclusive).
     * @param end Ending column index of the column to be extracted (exclusive)
     * @return Returns the specified column range from this sparse matrix.
     */
    public static <T extends Field<T>> AbstractCooFieldVector<?, ?, ?, ?, T> getRow(
            AbstractCooFieldMatrix<?, ?, ?, T> src,
            int rowIdx, int start, int end) {
        ValidateParameters.ensureIndicesInBounds(src.numRows, rowIdx);
        ValidateParameters.ensureIndicesInBounds(src.numCols, start, end-1);
        ValidateParameters.ensureLessEq(end-1, start);

        List<T> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i = 0; i<src.data.length; i++) {
            if(src.rowIndices[i]==rowIdx && src.colIndices[i] >= start && src.colIndices[i] < end) {
                entries.add(src.data[i]);
                indices.add(src.colIndices[i]);
            }
        }

        return src.makeLikeVector(new Shape(end-start),
                entries.toArray(src.makeEmptyDataArray(entries.size())),
                ArrayUtils.fromIntegerList(indices));
    }


    /**
     * Gets a specified column from this sparse matrix.
     * @param src Source sparse matrix to extract column from.
     * @param colIdx Index of the column to extract from the {@code src} matrix.
     * @return Returns the specified column from this sparse matrix.
     */
    public static <T extends Field<T>> AbstractCooFieldVector<?, ?, ?, ?, T> getCol(AbstractCooFieldMatrix<?, ?, ?, T> src, int colIdx) {
        ValidateParameters.ensureIndicesInBounds(src.numCols, colIdx);

        List<T> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i = 0; i<src.data.length; i++) {
            if(src.colIndices[i]==colIdx) {
                entries.add(src.data[i]);
                indices.add(src.rowIndices[i]);
            }
        }

        return src.makeLikeVector(new Shape(src.numRows),
                entries.toArray(src.makeEmptyDataArray(entries.size())),
                ArrayUtils.fromIntegerList(indices));
    }


    /**
     * Gets a specified column range from this sparse matrix.
     * @param src Source sparse matrix to extract column from.
     * @param colIdx Index of the column to extract from the {@code src} matrix.
     * @param start Staring row index of the column to be extracted (inclusive).
     * @param end Ending row index of the column to be extracted (exclusive)
     * @return Returns the specified column range from this sparse matrix.
     */
    public static <T extends Field<T>> AbstractCooFieldVector<?, ?, ?, ?, T> getCol(
            AbstractCooFieldMatrix<?, ?, ?, T> src,
            int colIdx, int start, int end) {
        ValidateParameters.ensureIndicesInBounds(src.numCols, colIdx);
        ValidateParameters.ensureIndicesInBounds(src.numRows, start, end);
        ValidateParameters.ensureLessEq(end, start);

        List<T> entries = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();

        for(int i = 0; i<src.data.length; i++) {
            if(src.colIndices[i]==colIdx && src.rowIndices[i] >= start && src.rowIndices[i] < end) {
                entries.add(src.data[i]);
                indices.add(src.rowIndices[i]);
            }
        }

        return src.makeLikeVector(new Shape(end-start),
                (T[]) entries.toArray(new Field[entries.size()]),
                ArrayUtils.fromIntegerList(indices));
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
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V> getSlice(
            AbstractCooFieldMatrix<?, ?, ?, V> src,
            int rowStart, int rowEnd, int colStart, int colEnd) {
        ValidateParameters.ensureIndicesInBounds(src.numRows, rowStart, rowEnd-1);
        ValidateParameters.ensureIndicesInBounds(src.numCols, colStart, colEnd-1);

        Shape shape = new Shape(rowEnd-rowStart, colEnd-colStart);
        List<V> entries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        int start = SparseElementSearch.matrixBinarySearch(src.rowIndices, src.colIndices, rowStart, colStart);

        if(start < 0) {
            // If no item with the specified indices is found, then begin search at the insertion point.
            start = -start - 1;
        }

        for(int i = start; i<src.data.length; i++) {
            if(inSlice(src.rowIndices[i], src.colIndices[i], rowStart, rowEnd, colStart, colEnd)) {
                entries.add(src.data[i]);
                rowIndices.add(src.rowIndices[i]-rowStart);
                colIndices.add(src.colIndices[i]-colStart);
            }
        }

        return src.makeLikeTensor(shape, entries, rowIndices, colIndices);
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
    private static <V extends Field<V>> void copyValuesNotInSlice(
            AbstractCooFieldMatrix<?, ?, ?, V> src, List<V> entries,
            List<Integer> rowIndices,
            List<Integer> colIndices, int[] rowRange, int[] colRange) {
        // Copy values not in slice.
        for(int i = 0; i<src.data.length; i++) {
            if( !(ArrayUtils.contains(rowRange, src.rowIndices[i])
                    && ArrayUtils.contains(colRange, src.colIndices[i])) ) {
                // Then the entry is not in the slice so add it.
                entries.add(src.data[i]);
                rowIndices.add(src.rowIndices[i]);
                colIndices.add(src.colIndices[i]);
            }
        }
    }


    /**
     * Checks that parameters are valued for setting a slice of a matrix.
     * @param src Matrix to set slice of.
     * @param values Matrix representing slice.
     * @param row Starting row for slice.
     * @param col Ending row for slice.
     */
    private static <T extends MatrixMixin<?, ?, ?, ?>, U extends MatrixMixin<?, ?, ?, ?>> void setSliceParamCheck(
            T src, U values, int row, int col) {
        
        ValidateParameters.ensureIndicesInBounds(src.numRows(), row);
        ValidateParameters.ensureIndicesInBounds(src.numCols(), col);
        ValidateParameters.ensureLessEq(src.numRows(), values.numRows() + row);
        ValidateParameters.ensureLessEq(src.numCols(), values.numCols() + col);
    }


    /**
     * Checks that parameters are valued for setting a slice of a matrix.
     * @param src Matrix to set slice of.
     * @param valueRows Number of rows in slice to set.
     * @param valueCols Number of columns in slice to be set.
     * @param row Starting row for slice.
     * @param col Ending row for slice.
     */
    private static <T extends MatrixMixin<?, ?, ?, ?>> void setSliceParamCheck(
            T src, int valueRows, int valueCols, int row, int col) {
        ValidateParameters.ensureIndicesInBounds(src.numRows(), row);
        ValidateParameters.ensureIndicesInBounds(src.numCols(), col);
        ValidateParameters.ensureLessEq(src.numRows(), valueRows + row);
        ValidateParameters.ensureLessEq(src.numCols(), valueCols + col);
    }
}
