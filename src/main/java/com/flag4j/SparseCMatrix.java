/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.SparseMatrixBase;
import com.flag4j.util.ArrayUtils;

/**
 * Complex sparse matrix. Stored in COO (Coordinate) format.
 */
public class SparseCMatrix extends SparseMatrixBase<CNumber[]> {


    /**
     * Creates a square sparse matrix filled with zeros.
     * @param size size of the square matrix.
     */
    public SparseCMatrix(int size) {
        super(new Shape(size, size), 0, new CNumber[0], new int[0], new int[0]);
    }


    /**
     * Creates a sparse matrix of specified size filled with zeros.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     */
    public SparseCMatrix(int rows, int cols) {
        super(new Shape(rows, cols), 0, new CNumber[0], new int[0], new int[0]);
    }


    /**
     * Creates a sparse matrix of specified shape filled with zeros.
     * @param shape Shape of this sparse matrix.
     */
    public SparseCMatrix(Shape shape) {
        super(shape, 0, new CNumber[0], new int[0], new int[0]);
    }


    /**
     * Creates a  square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseCMatrix(int size, CNumber[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size), nonZeroEntries.length, nonZeroEntries, rowIndices, colIndices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse matrix with specified size, non-zero entries, row indices, and column indices.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseCMatrix(int rows, int cols, CNumber[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), nonZeroEntries.length, nonZeroEntries, rowIndices, colIndices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param shape Shape of the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseCMatrix(Shape shape, CNumber[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape, nonZeroEntries.length, nonZeroEntries, rowIndices, colIndices);
    }


    /**
     * Creates a  square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseCMatrix(int size, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size), nonZeroEntries.length, new CNumber[nonZeroEntries.length], rowIndices, colIndices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse matrix with specified size, non-zero entries, row indices, and column indices.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseCMatrix(int rows, int cols, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), nonZeroEntries.length, new CNumber[nonZeroEntries.length], rowIndices, colIndices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param shape Shape of the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseCMatrix(Shape shape, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape, nonZeroEntries.length, new CNumber[nonZeroEntries.length], rowIndices, colIndices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a  square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseCMatrix(int size, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size), nonZeroEntries.length, new CNumber[nonZeroEntries.length], rowIndices, colIndices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse matrix with specified size, non-zero entries, row indices, and column indices.
     * @param rows Number of rows in the sparse matrix.
     * @param cols Number of columns in the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseCMatrix(int rows, int cols, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), nonZeroEntries.length, new CNumber[nonZeroEntries.length], rowIndices, colIndices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param shape Shape of the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     * @throws IllegalArgumentException If the number of non-zero entries does not fit within the given shape. Or, if the
     * lengths of the non-zero entries, row indices, and column indices arrays are not all the same.
     */
    public SparseCMatrix(Shape shape, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape, nonZeroEntries.length, new CNumber[nonZeroEntries.length], rowIndices, colIndices);
        ArrayUtils.copy2CNumber(nonZeroEntries, super.entries);
    }


//    /**
//     * Creates a sparse matrix whose entries are specified by a 2D array. This is not the recommended method of constructing
//     * a sparse matrix. It is recommended to use {@link #SparseCMatrix(Shape, double[], int[], int[])} if the coordinate information is already known.
//     * @param entries Entries of sparse matrix.
//     */
//    public SparseCMatrix(double[][] entries) {
//        super(new Shape(entries.length, entries[0].length));
//
//        ArrayList<Double> nonZeroEntries = new ArrayList<>(this.totalEntries()/8);
//        ArrayList<Integer> rowIndices = new ArrayList<>(this.totalEntries()/8);
//        ArrayList<Integer> colIndices = new ArrayList<>(this.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        int numCols=super.numCols();
//        for(int i=0; i<entries.length; i++) {
//            for(int j=0; j<entries[0].length; j++) {
//                if(entries[i][j] != 0) {
//                    nonZeroEntries.add(entries[i][j]);
//                    rowIndices.add(i);
//                    colIndices.add(j);
//                }
//            }
//        }
//
//        nonZeroEntries.trimToSize();
//
//        super.setNonZeroEntries(nonZeroEntries.size());
//        super.entries = new CNumber[super.nonZeroEntries()];
//
//        for(int i=0; i<nonZeroEntries.size(); i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
//
//        super.rowIndices = rowIndices.stream().mapToInt(Integer::intValue).toArray();
//        super.colIndices = colIndices.stream().mapToInt(Integer::intValue).toArray();
//
//    }
//
//
//    /**
//     * Creates a sparse matrix whose entries are specified by a 2D array. This is not the recommended method of constructing
//     * a sparse matrix. It is recommended to use {@link #SparseCMatrix(Shape, int[], int[], int[])} if the coordinate information is already known.
//     * @param entries Entries of sparse matrix.
//     */
//    public SparseCMatrix(int[][] entries) {
//        super(new Shape(entries.length, entries[0].length));
//
//        ArrayList<Integer> nonZeroEntries = new ArrayList<>(this.totalEntries()/8);
//        ArrayList<Integer> rowIndices = new ArrayList<>(this.totalEntries()/8);
//        ArrayList<Integer> colIndices = new ArrayList<>(this.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            for(int j=0; j<entries[0].length; j++) {
//                if(entries[i][j] != 0) {
//                    nonZeroEntries.add(entries[i][j]);
//                    rowIndices.add(i);
//                    colIndices.add(j);
//                }
//            }
//        }
//
//        nonZeroEntries.trimToSize();
//        super.setNonZeroEntries(nonZeroEntries.size());
//        super.entries = new CNumber[super.nonZeroEntries()];
//
//        for(int i=0; i<nonZeroEntries.size(); i++) {
//            super.entries[i] = new CNumber(nonZeroEntries.get(i));
//        }
//
//        super.rowIndices = rowIndices.stream().mapToInt(Integer::intValue).toArray();
//        super.colIndices = colIndices.stream().mapToInt(Integer::intValue).toArray();
//    }
//
//
//    /**
//     * Creates a sparse matrix whose entries are specified by a 2D array. This is not the recommended method of constructing
//     * a sparse matrix. It is recommended to use {@link #SparseCMatrix(Shape, CNumber[], int[], int[])} if the coordinate information is already known.
//     * @param entries Entries of sparse matrix.
//     */
//    public SparseCMatrix(CNumber[][] entries) {
//        super(new Shape(entries.length, entries[0].length));
//
//        ArrayList<CNumber> nonZeroEntries = new ArrayList<>(this.totalEntries()/8);
//        ArrayList<Integer> rowIndices = new ArrayList<>(this.totalEntries()/8);
//        ArrayList<Integer> colIndices = new ArrayList<>(this.totalEntries()/8);
//
//        // Fill entries with non-zero values.
//        for(int i=0; i<entries.length; i++) {
//            for(int j=0; j<entries[0].length; j++) {
//                if(entries[i][j].re != 0 && entries[i][j].im != 0) {
//                    nonZeroEntries.add(entries[i][j]);
//                    rowIndices.add(i);
//                    colIndices.add(j);
//                }
//            }
//        }
//
//        nonZeroEntries.trimToSize();
//        super.setNonZeroEntries(nonZeroEntries.size());
//        super.entries = new CNumber[super.nonZeroEntries()];
//
//        for(int i=0; i<nonZeroEntries.size(); i++) {
//            super.entries[i] = nonZeroEntries.get(i).clone();
//        }
//
//        super.rowIndices = rowIndices.stream().mapToInt(Integer::intValue).toArray();
//        super.colIndices = colIndices.stream().mapToInt(Integer::intValue).toArray();
//    }


    /**
     * Constructs a sparse complex matrix whose non-zero entries, indices, and shape are specified by another
     * complex sparse matrix.
     * @param A Complex sparse matrix to copy.
     */
    public SparseCMatrix(SparseCMatrix A) {
        super(A.shape.clone(), A.nonZeroEntries(), new CNumber[A.nonZeroEntries()], A.rowIndices.clone(), A.colIndices.clone());
        ArrayUtils.copy2CNumber(A.entries, super.entries);
    }
}
