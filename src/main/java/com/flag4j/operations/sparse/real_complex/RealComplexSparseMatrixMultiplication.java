/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.operations.sparse.real_complex;

import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.concurrency.ThreadManager;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;


/**
 * This class contains low level methods for computing the multiplication between a real/complex matrix and a complex/real
 * matrix/vector. <br>
 * <b>WARNING:</b> The methods in this class do not provide sanity checks.
 */
public class RealComplexSparseMatrixMultiplication {

    private RealComplexSparseMatrixMultiplication() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
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
                                     double[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {

        int rows1 = shape1.dims[0];
        int cols2 = shape2.dims[1];

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fillZeros(dest);

        // r1, c1, r2, and c2 store row/column indices for non-zero values in src1 and src2.
        int r1, c1, r2, c2;

        for(int i=0; i<src1.length; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                r2 = rowIndices2[j]; // = k
                c2 = colIndices2[j]; // = j

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1*cols2 + c2].addEq(src1[i].mult(src2[j]));
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a concurrent implementation of
     * the standard algorithm.
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
                                               double[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {

        int rows1 = shape1.dims[0];
        int cols2 = shape2.dims[1];

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fillZeros(dest);

        ThreadManager.concurrentLoop(0, src1.length, (i)->{
            int r1 = rowIndices1[i]; // = i
            int c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                int r2 = rowIndices2[j]; // = k
                int c2 = colIndices2[j]; // = j

                if(c1==r2) { // Then we multiply and add to sum.
                    CNumber product = src1[i].mult(src2[j]);

                    synchronized (dest) {
                        dest[r1*cols2 + c2].addEq(product);
                    }
                }
            }
        });

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
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static CNumber[] standardVector(CNumber[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                           double[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.dims[0];
        int cols2 = shape2.dims[1];

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fillZeros(dest);

        // r1, c1, and r2 store the indices for non-zero values in src1 and src2.
        int r1, c1, r2;

        for(int i=0; i<src1.length; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1*cols2].addEq(src1[i].mult(src2[j]));
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
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static CNumber[] concurrentStandardVector(CNumber[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                                     double[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.dims[0];
        int cols2 = shape2.dims[1];

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fillZeros(dest);

        ThreadManager.concurrentLoop(0, src1.length, (i) -> {
            int r1 = rowIndices1[i]; // = i
            int c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                int r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    CNumber product = src1[i].mult(src2[j]);

                    synchronized (dest) {
                        dest[r1*cols2].addEq(product);
                    }
                }
            }
        });

        return dest;
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
    public static CNumber[] standard(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                     CNumber[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {

        int rows1 = shape1.dims[0];
        int cols2 = shape2.dims[1];

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fillZeros(dest);

        // r1, c1, r2, and c2 store row/column indices for non-zero values in src1 and src2.
        int r1, c1, r2, c2;

        for(int i=0; i<src1.length; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                r2 = rowIndices2[j]; // = k
                c2 = colIndices2[j]; // = j

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1*cols2 + c2].addEq(src2[j].mult(src1[i]));
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two sparse matrices using a concurrent implementation of
     * the standard algorithm.
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
    public static CNumber[] concurrentStandard(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                               CNumber[] src2, int[] rowIndices2, int[] colIndices2, Shape shape2) {

        int rows1 = shape1.dims[0];
        int cols2 = shape2.dims[1];

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fillZeros(dest);

        ThreadManager.concurrentLoop(0, src1.length, (i)->{
            int r1 = rowIndices1[i]; // = i
            int c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                int r2 = rowIndices2[j]; // = k
                int c2 = colIndices2[j]; // = j

                if(c1==r2) { // Then we multiply and add to sum.
                    CNumber product = src2[j].mult(src1[i]);

                    synchronized (dest) {
                        dest[r1*cols2 + c2].addEq(product);
                    }
                }
            }
        });

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
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static CNumber[] standardVector(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                           CNumber[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.dims[0];
        int cols2 = shape2.dims[1];

        CNumber[] dest = new CNumber[rows1*cols2];

        // r1, c1, r2, and store the indices for non-zero values in src1 and src2.
        int r1, c1, r2;

        for(int i=0; i<src1.length; i++) {
            r1 = rowIndices1[i]; // = i
            c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    dest[r1*cols2].addEq(src2[j].mult(src1[i]));
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
     * @param shape2 Shape of the second sparse matrix.
     * @return The result of the matrix-vector multiplication stored in a dense matrix.
     */
    public static CNumber[] concurrentStandardVector(double[] src1, int[] rowIndices1, int[] colIndices1, Shape shape1,
                                                     CNumber[] src2, int[] indices, Shape shape2) {

        int rows1 = shape1.dims[0];
        int cols2 = shape2.dims[1];

        CNumber[] dest = new CNumber[rows1*cols2];

        ThreadManager.concurrentLoop(0, src1.length, (i) -> {
            int r1 = rowIndices1[i]; // = i
            int c1 = colIndices1[i]; // = k

            for(int j=0; j<src2.length; j++) {
                int r2 = indices[j]; // = k

                if(c1==r2) { // Then we multiply and add to sum.
                    CNumber product = src2[j].mult(src1[i]);

                    synchronized (dest) {
                        dest[r1*cols2].addEq(product);
                    }
                }
            }
        });

        return dest;
    }
}
