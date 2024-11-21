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

package org.flag4j.linalg.operations.sparse.coo.semiring_ops;

import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.linalg.operations.sparse.SparseUtils;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * This utility class contains methods useful for computing the matrix multiplication between two sparse COO
 * {@link Semiring} tensors.
 */
public final class CooSemiringMatMult {

    private CooSemiringMatMult() {
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(getClass()));
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a standard algorithm.
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero entries for the second sparse matrix.
     * @param colIndices2 column indices of non-zero entries for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static <T extends Semiring<T>> void standard(
            Semiring<T>[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
            Semiring<T>[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2,
            Semiring<T>[] dest) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, src1[0].getZero());

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        for(int i=0; i<src1.length; i++) {
            int c1 = colIndices1[i]; // = k
            Semiring<T> value = src1[i];

            // Check if any values in src2 have the same row index as the column index of the value in src1.
            if(map.containsKey(c1)) {
                int r1 = rowIndices1[i]; // = i
                int rowIdx = r1*cols2;

                for(int j : map.get(c1)) { // Iterate over all entries in src2 where rowIndices[j] == colIndices[j]
                    int c2 = colIndices2[j]; // = j
                    dest[rowIdx + c2] = dest[rowIdx + c2].add(value.mult((T) src2[j]));
                }
            }
        }
    }


    /**
     * <p>Computes the matrix multiplication between two sparse matrices using a concurrent implementation of
     * the standard algorithm.</p>
     *
     * <p>NOTE: Caution should be exercised when using this method.
     * It is rarely faster than {@link #standard(Semiring[], int[], int[], Shape, Semiring[], int[], int[], Shape, Semiring[])}</p>
     *
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param rowIndices2 Row indices of non-zero entries for the second sparse matrix.
     * @param colIndices2 column indices of non-zero entries for the second sparse matrix.
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix multiplication stored in a dense matrix.
     */
    public static <T extends Semiring<T>> Semiring<T>[] concurrentStandard(
            Semiring<T>[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
            Semiring<T>[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);

        final Semiring<T> ZERO = src1[0].getZero();

        Semiring<T>[] dest = new Semiring[rows1*cols2];
        Arrays.fill(dest, ZERO);
        ConcurrentMap<Integer, Semiring<T>> destMap = new ConcurrentHashMap<>();

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int c1 = colIndices1[i]; // = k

                // Check if any values in src2 have the same row index as the column index of the value in src1.
                if(map.containsKey(c1)) {
                    int r1 = rowIndices1[i]; // = i
                    int rowIdx = r1*cols2;
                    Semiring<T> value = src1[i];

                    for(int j : map.get(c1)) { // Iterate over all entries in src2 where rowIndices[j] == colIndices[j]
                        int idx = rowIdx + colIndices2[j];
                        destMap.put(idx, destMap.getOrDefault(idx, ZERO).add(value.mult((T) src2[j])));
                    }
                }
            }
        });

        // Copy values from map to destination array.
        for(int idx : destMap.keySet()) {
            dest[idx] = destMap.get(idx);
        }

        return dest;
    }


    /**
     * <p>Computes the multiplication between a sparse matrix and a sparse vector using a standard algorithm.
     *
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param dest Array to store the resulting dense vector in.
     */
    public static <T extends Semiring<T>> void standardVector(
            Semiring<T>[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
            Semiring<T>[] src2, int[] indices, Semiring<T>[] dest) {

        final Semiring<T> ZERO = src1[0].getZero();
        int rows1 = shape1.get(0);
        Arrays.fill(dest, ZERO);

        // r1, c1, r2, and store the indices for non-zero values in src1 and src2.
        int r1, c1, r2;

        for(int i=0, size1=src1.length; i<size1; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k
            Semiring<T> sum = ZERO;

            for(int j=0, size2=src2.length; j<size2; j++) {
                r2 = indices[j]; // = k

                if(c1==r2) // Then multiply and add to sum.
                    sum = sum.add(src1[i].mult((T) src2[j]));
            }

            dest[r1] = sum;
        }
    }


    /**
     * <p>Computes the multiplication between a sparse matrix and a sparse vector using a concurrent implementation
     * of the standard algorithm.</p>
     *
     * <p>NOTE: Caution should be exercised when using this method.
     * It is rarely faster than {@link #standardVector(Semiring[], int[], int[], Shape, Semiring[], int[])}</p>
     * 
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static <T extends Semiring<T>> Semiring<T>[] concurrentStandardVector(
            Semiring<T>[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
            Semiring<T>[] src2, int[] indices) {
        int rows1 = shape1.get(0);
        Semiring<T>[] dest = new Semiring[rows1];
        Arrays.fill(dest, src1[0].getZero());

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int r1 = rowIndices1[i]; // = i
                int c1 = colIndices1[i]; // = k

                for(int j=0; j<src2.length; j++) {
                    int r2 = indices[j]; // = k

                    if(c1==r2) { // Then we multiply and add to sum.
                        T product = src1[i].mult((T) src2[j]);

                        synchronized (dest) {
                            dest[r1] = dest[r1].add(product);
                        }
                    }
                }
            }
        });

        return dest;
    }
}
