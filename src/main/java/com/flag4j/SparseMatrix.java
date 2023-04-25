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


import com.flag4j.core.RealMatrixBase;
import com.flag4j.core.RealSparseMatrixBase;

import java.util.Arrays;

/**
 * Real sparse matrix. Matrix is stored in coordinate list (COO) format.
 * TODO: Temporarily abstract
 */
public abstract class SparseMatrix extends RealSparseMatrixBase {


    /**
     * Creates a square sparse matrix of specified size filled with zeros.
     * @param size The number of rows/columns in this sparse matrix.
     */
    public SparseMatrix(int size) {
        super(new Shape(size, size), 0, new double[0], new int[0], new int[0]);
    }


    /**
     * Creates a sparse matrix of specified number of rows and columns filled with zeros.
     * @param rows The number of rows in this sparse matrix.
     * @param cols The number of columns in this sparse matrix.
     */
    public SparseMatrix(int rows, int cols) {
        super(new Shape(rows, cols), 0, new double[0], new int[0], new int[0]);
    }


    /**
     * Creates a sparse matrix of specified shape filled with zeros.
     * @param shape Shape of this sparse matrix.
     */
    public SparseMatrix(Shape shape) {
        super(shape, 0, new double[0], new int[0], new int[0]);
    }


    /**
     * Creates a square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(int size, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size), nonZeroEntries.length, nonZeroEntries, rowIndices, colIndices);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param rows The number of rows in this sparse matrix.
     * @param cols The number of columns in this sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(int rows, int cols, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), nonZeroEntries.length, nonZeroEntries, rowIndices, colIndices);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param shape Shape of the sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(Shape shape, double[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape, nonZeroEntries.length, nonZeroEntries, rowIndices, colIndices);
    }


    /**
     * Creates a square sparse matrix with specified non-zero entries, row indices, and column indices.
     * @param size Size of the square matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(int size, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size), nonZeroEntries.length, Arrays.stream(nonZeroEntries).asDoubleStream().toArray(),
                rowIndices, colIndices);
    }


    /**
     * Creates a sparse matrix with specified shape, non-zero entries, row indices, and column indices.
     * @param rows The number of rows in this sparse matrix.
     * @param cols The number of columns in this sparse matrix.
     * @param nonZeroEntries Non-zero entries of sparse matrix.
     * @param rowIndices Row indices of the non-zero entries.
     * @param colIndices Column indices of the non-zero entries.
     */
    public SparseMatrix(int rows, int cols, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), nonZeroEntries.length, Arrays.stream(nonZeroEntries).asDoubleStream().toArray(),
                rowIndices, colIndices);
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
    public SparseMatrix(Shape shape, int[] nonZeroEntries, int[] rowIndices, int[] colIndices) {
        super(shape, nonZeroEntries.length, Arrays.stream(nonZeroEntries).asDoubleStream().toArray(),
                rowIndices, colIndices);
    }


//    /**
//     * Creates a sparse matrix whose entries are specified by a 2D array. This is not the recommended method of constructing
//     * a sparse matrix. It is recommended to use {@link #SparseMatrix(Shape, double[], int[], int[])} if the coordinate information is already known.
//     * @param entries Entries of sparse matrix.
//     */
//    public SparseMatrix(double[][] entries) {
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
//        super.entries = nonZeroEntries.stream().mapToDouble(Double::doubleValue).toArray();
//        super.rowIndices = rowIndices.stream().mapToInt(Integer::intValue).toArray();
//        super.colIndices = colIndices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }
//
//
//    /**
//     * Creates a sparse matrix whose entries are specified by a 2D array. This is not the recommended method of constructing
//     * a sparse matrix. It is recommended to use {@link #SparseMatrix(Shape, int[], int[], int[])} if the coordinate information is already known.
//     * @param entries Entries of sparse matrix.
//     */
//    public SparseMatrix(int[][] entries) {
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
//        super.entries = nonZeroEntries.stream().mapToDouble(Integer::doubleValue).toArray();
//        super.rowIndices = rowIndices.stream().mapToInt(Integer::intValue).toArray();
//        super.colIndices = colIndices.stream().mapToInt(Integer::intValue).toArray();
//        super.setNonZeroEntries(super.entries.length);
//    }

    /**
     * Constructs a sparse tensor whose shape and values are given by another sparse tensor. This effectively copies
     * the tensor.
     * @param A Sparse Matrix to copy.
     */
    public SparseMatrix(SparseMatrix A) {
        super(A.shape.copy(), A.nonZeroEntries(), A.entries.clone(), A.rowIndices.clone(), A.colIndices.clone());
    }


    @Override
    public SparseMatrix flatten(int axis) {
        return null;
    }
}
