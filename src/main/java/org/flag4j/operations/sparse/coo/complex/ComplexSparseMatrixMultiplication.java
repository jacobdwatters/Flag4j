/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.operations.sparse.coo.complex;

import org.flag4j.arrays.Shape;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.operations.sparse.SparseUtils;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * This class contains low level methods for computing the matrix multiplication of sparse complex matrices/vectors.<br>
 * <b>WARNING:</b> The methods in this class do not perform any sanity checks.
 */
public final class ComplexSparseMatrixMultiplication {

    private ComplexSparseMatrixMultiplication() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
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
    public static CNumber[] standard(CNumber[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                     CNumber[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        Arrays.fill(dest, CNumber.ZERO);

        // Create a map where key is row index from src2.
        // and value is a list of indices in src2 where this row appears.
        Map<Integer, List<Integer>> map = SparseUtils.createMap(src2.length, rowIndices2);

        for(int i=0; i<src1.length; i++) {
            int c1 = colIndices1[i]; // = k

            // Check if any values in src2 have the same row index as the column index of the value in src1.
            if(map.containsKey(c1)) {
                int r1 = rowIndices1[i]; // = i
                int rowIdx = r1*cols2;

                for(int j : map.get(c1)) { // Iterate over all entries in src2 where rowIndices[j] == colIndices[j]
                    int c2 = colIndices2[j]; // = j
                    dest[rowIdx + c2] = dest[rowIdx + c2].add(src1[i].mult(src2[j]));
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a concurrent implementation of
     * the standard algorithm. <br><br>
     * 
     * NOTE: Caution should be exercised when using this method. It is rarely faster than {@link #standard(CNumber[], int[], int[], Shape, CNumber[], int[], int[], Shape)}
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
    public static CNumber[] concurrentStandard(CNumber[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                               CNumber[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        Arrays.fill(dest, CNumber.ZERO);
        ConcurrentMap<Integer, CNumber> destMap = new ConcurrentHashMap<>();

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

                    for(int j : map.get(c1)) { // Iterate over all entries in src2 where rowIndices[j] == colIndices[j]
                        int idx = rowIdx + colIndices2[j];
                        destMap.put(idx, destMap.getOrDefault(idx, CNumber.ZERO).add(src1[i].mult(src2[j])));
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
     * Computes the multiplication between a sparse matrix and a sparse vector using a standard algorithm.
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static CNumber[] standardVector(CNumber[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                           CNumber[] src2, int[] indices) {

        int rows1 = shape1.get(0);

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        // r1, c1, r2, and store the indices for non-zero values in src1 and src2.
        int r1, c1, r2;

        for(int i=0; i<src1.length; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1] = dest[r1].add(src1[i].mult(src2[j]));
                }
            }
        }

        return dest;
    }


    /**
     * Computes the multiplication between a sparse matrix and a sparse vector using a concurrent implementation
     * of the standard algorithm.
     * @param src1 Non-zero entries of the first sparse matrix.
     * @param rowIndices1 Row indices of non-zero entries for the first sparse matrix.
     * @param colIndices1 Column indices of non-zero entries for the first sparse matrix.
     * @param shape1 Shape of the first sparse matrix.
     * @param src2 Non-zero entries of the second sparse matrix.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static CNumber[] concurrentStandardVector(CNumber[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                                     CNumber[] src2, int[] indices) {
        int rows1 = shape1.get(0);

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int r1 = rowIndices1[i]; // = i
                int c1 = colIndices1[i]; // = k

                for(int j=0; j<src2.length; j++) {
                    int r2 = indices[j]; // = k

                    if(c1==r2) { // Then we multiply and add to sum.
                        CNumber product = src1[i].mult(src2[j]);

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
