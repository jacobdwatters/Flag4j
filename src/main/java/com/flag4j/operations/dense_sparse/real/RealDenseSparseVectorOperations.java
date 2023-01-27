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

package com.flag4j.operations.dense_sparse.real;


import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class provides low level methods for computing operations between a real dense/sparse vector and a
 * real sparse/dense vector.
 */
public class RealDenseSparseVectorOperations {


    private RealDenseSparseVectorOperations() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the vector inner product between a real dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices
     * @param sparseSize
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of entries in the two vectors is not equivalent.
     */
    public static double innerProduct(double[] src1, double[] src2, int[] indices, int sparseSize) {
        ParameterChecks.assertArrayLengthsEq(src1.length, sparseSize);
        double innerProd = 0;
        int index;

        for(int i=0; i<src2.length; i++) {
            index = indices[i];

            innerProd += src1[index]*src2[i];
        }

        return innerProd;
    }


    /**
     * Computes the vector outer product between a real dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries of sparse vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static double[] outerProduct(double[] src1, double[] src2, int[] indices, int sparseSize) {
        double[] dest = new double[src1.length*sparseSize];
        int index;

        for(int i=0; i<src1.length; i++) {
            for(int j=0; j<src2.length; j++) {
                index = indices[j];

                dest[i*src2.length + index] = src1[i]*src2[j];
            }
        }

        return dest;
    }
}