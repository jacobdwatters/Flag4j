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
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldMatrix;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.sparse.CooFieldMatrix;
import org.flag4j.arrays.sparse.CooFieldVector;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class has low level implementations for ops between two sparse coo
 * {@link Field} matrices.
 */
public final class CooFieldMatrixOps {

    private CooFieldMatrixOps() {
        // Hide default constructor for utility class.
    }

    /**
     * Adds two real sparse matrices. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the sum.
     * @param src2 Second matrix in the sum.
     * @return The sum of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V>
    add(AbstractCooFieldMatrix<?, ?, ?, V> src1, AbstractCooFieldMatrix<?, ?, ?, V> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);

        List<V> sum = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1.data.length || src2Counter < src2.data.length) {

            if(src1Counter >= src1.data.length || src2Counter >= src2.data.length) {
                add1 = src2Counter >= src2.data.length;
                add2 = !add1;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1.colIndices[src1Counter] < src2.colIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                sum.add(src1.data[src1Counter].add(src2.data[src2Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                sum.add(src1.data[src1Counter]);
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
            } else {
                sum.add(src2.data[src2Counter]);
                rowIndices.add(src2.rowIndices[src2Counter]);
                colIndices.add(src2.colIndices[src2Counter]);
                src2Counter++;
            }
        }

        return src1.makeLikeTensor(src1.shape, sum, rowIndices, colIndices);
    }


    /**
     * Adds a double all data (including zero values) of a real sparse matrix.
     * @param src Sparse matrix to add double value to.
     * @param a Double value to add to the sparse matrix.
     * @return The result of the matrix addition.
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} data in the matrix (including zero data).
     */
    public static <V extends Field<V>> FieldMatrix<V>
    add(AbstractCooFieldMatrix<?, ?, ?, V> src, double a) {
        V[] sum = (V[]) new Field[src.totalEntries().intValueExact()];
        Arrays.fill(sum, a);

        int row;
        int col;

        for(int i = 0; i<src.data.length; i++) {
            row = src.rowIndices[i];
            col = src.colIndices[i];
            sum[row*src.numCols + col] = sum[row*src.numCols + col].add(src.data[i]);
        }

        return new FieldMatrix<V>(src.shape, sum);
    }


    /**
     * Computes the subtraction between two real sparse matrices. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the subtraction.
     * @param src2 Second matrix in the subtraction.
     * @return The difference of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V>
    sub(AbstractCooFieldMatrix<?, ?, ?, V> src1, AbstractCooFieldMatrix<?, ?, ?, V> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);

        List<V> sum = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1.data.length || src2Counter < src2.data.length) {

            if(src1Counter >= src1.data.length || src2Counter >= src2.data.length) {
                add1 = src2Counter >= src2.data.length;
                add2 = !add1;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1.colIndices[src1Counter] < src2.colIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                sum.add(src1.data[src1Counter].sub(src2.data[src2Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                sum.add(src1.data[src1Counter]);
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
            } else {
                sum.add(src2.data[src2Counter].addInv());
                rowIndices.add(src2.rowIndices[src2Counter]);
                colIndices.add(src2.colIndices[src2Counter]);
                src2Counter++;
            }
        }

        return src1.makeLikeTensor(src1.shape, sum, rowIndices, colIndices);
    }


    /**
     * Subtracts a double from all data (including zero values) of a real sparse matrix.
     * @param src Sparse matrix to subtract double value from.
     * @param a Double value to subtract from the sparse matrix.
     * @return The result of the matrix subtraction.
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} data in the matrix (including zero data).
     */
    public static <V extends Field<V>> FieldMatrix<V>
    sub(AbstractCooFieldMatrix<?, ?, ?, V> src, double a) {
        V[] sum = (V[]) new Field[src.totalEntries().intValueExact()];
        Arrays.fill(sum, -a);

        int row;
        int col;

        for(int i = 0; i<src.data.length; i++) {
            row = src.rowIndices[i];
            col = src.colIndices[i];
            sum[row*src.numCols + col] = sum[row*src.numCols + col].add((V) src.data[i]);
        }

        return new FieldMatrix<V>(src.shape, sum);
    }



    /**
     * Multiplies two sparse matrices element-wise. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the element-wise multiplication.
     * @param src2 Second matrix in the element-wise multiplication.
     * @return The element-wise product of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static <V extends Field<V>> AbstractCooFieldMatrix<?, ?, ?, V>
    elemMult(AbstractCooFieldMatrix<?, ?, ?, V> src1, AbstractCooFieldMatrix<?, ?, ?, V> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);

        List<V> product = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                product.add(src1.data[src1Counter].mult(src2.data[src2Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.

                if(src1.colIndices[src1Counter] < src2.colIndices[src2Counter]) {
                    src1Counter++;
                } else {
                    src2Counter++;
                }
            } else {
                if(src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter]) {
                    src1Counter++;
                } else {
                    src2Counter++;
                }
            }
        }

        return src1.makeLikeTensor(src1.shape, product, rowIndices, colIndices);
    }

    /**
     * Adds a sparse vector to each column of a sparse matrix as if the vector is a column vector.
     * @param src The source sparse matrix.
     * @param col Sparse vector to add to each column of the sparse matrix.
     * @return A dense copy of the {@code src} matrix with the {@code col} vector added to each row of the matrix.
     */
    public static <T extends Field<T>> FieldMatrix<T> addToEachCol(CooFieldMatrix<T> src, CooFieldVector<T> col) {
        ValidateParameters.ensureEquals(src.numRows, col.size);
        T[] destEntries = (T[]) new Field[src.totalEntries().intValueExact()];

        // Add values from sparse matrix.
        for(int i = 0; i<src.data.length; i++) {
            destEntries[src.rowIndices[i]*src.numCols + src.colIndices[i]] = src.data[i];
        }

        // Add values from sparse column.
        for(int i = 0; i<col.data.length; i++) {
            int idx = col.indices[i]*src.numCols;
            int end = idx + src.numCols;
            T value = (T) col.data[i];

            while(idx < end) {
                destEntries[idx] = destEntries[idx++].add(value);
            }
        }

        return new FieldMatrix<T>(src.shape, destEntries);
    }


    /**
     * Adds a sparse vector to each row of a sparse matrix as if the vector is a row vector.
     * @param src The source sparse matrix.
     * @param row Sparse vector to add to each row of the sparse matrix.
     * @return A dense copy of the {@code src} matrix with the {@code row} vector added to each row of the matrix.
     */
    public static <T extends Field<T>> FieldMatrix<T> addToEachRow(CooFieldMatrix<T> src, CooFieldVector<T> row) {
        ValidateParameters.ensureEquals(src.numCols, row.size);
        T[] destEntries = (T[]) new Field[src.totalEntries().intValueExact()];

        // Add values from sparse matrix.
        for(int i = 0; i<src.data.length; i++) {
            destEntries[src.rowIndices[i]*src.numCols + src.colIndices[i]] = src.data[i];
        }

        // Add values from sparse column.
        for(int i = 0; i<row.data.length; i++) {
            int colIdx = row.indices[i];
            int idx = colIdx;
            T value = row.data[i];

            while(idx < destEntries.length) {
                destEntries[idx] = destEntries[idx].add(value);
                idx += src.numCols;
            }
        }

        return new FieldMatrix<T>(src.shape, destEntries);
    }
}
