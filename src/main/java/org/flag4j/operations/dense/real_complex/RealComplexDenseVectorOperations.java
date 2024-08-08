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

package org.flag4j.operations.dense.real_complex;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This class provides low level implementations for vector operations with a real/complex dense vector and a complex/real
 * dense vector.
 */
public final class RealComplexDenseVectorOperations {


    private RealComplexDenseVectorOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the vector inner product for a real dense vector and a complex dense vector.
     * @param src1 Entries of the first vector.
     * @param src2 Entries of the second vector.
     * @return The inner product of the two vectors.
     */
    public static CNumber innerProduct(double[] src1, CNumber[] src2) {
        ParameterChecks.assertArrayLengthsEq(src1.length, src2.length);
        CNumber innerProd = CNumber.ZERO;

        for(int i=0; i<src1.length; i++) {
            innerProd = innerProd.add(src2[i].conj().mult(src1[i]));
        }

        return innerProd;
    }



    /**
     * Computes the vector inner product for a complex dense vector and a real dense vector.
     * @param src1 Entries of the first vector.
     * @param src2 Entries of the second vector.
     * @return The inner product of the two vectors.
     */
    public static CNumber innerProduct(CNumber[] src1, double[] src2) {
        ParameterChecks.assertArrayLengthsEq(src1.length, src2.length);
        CNumber innerProd = CNumber.ZERO;

        for(int i=0; i<src1.length; i++) {
            innerProd = innerProd.add(src1[i].mult(src2[i]));
        }

        return innerProd;
    }



    /**
     * Computes the vector outer product between two real dense vectors.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static CNumber[] outerProduct(double[] src1, CNumber[] src2) {
        int destIndex;
        CNumber[] dest = new CNumber[src1.length*src2.length];

        for(int i=0; i<src1.length; i++) {
            destIndex = i*src2.length;

            for(CNumber cNumber : src2) {
                dest[destIndex++] = cNumber.conj().mult(src1[i]);
            }
        }

        return dest;
    }


    /**
     * Computes the vector outer product between two real dense vectors.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static CNumber[] outerProduct(CNumber[] src1, double[] src2) {
        int destIndex;
        CNumber[] dest = new CNumber[src1.length*src2.length];

        for(int i=0; i<src1.length; i++) {
            destIndex = i*src2.length;

            for(double v : src2) {
                dest[destIndex++] = src1[i].mult(v);
            }
        }

        return dest;
    }

    /**
     * Adds a scalar value to all entries of a tensor.
     * @param src1 Entries of first tensor.
     * @param a Scalar to add to all entries of this tensor.
     * @return The result of adding the scalar to each entry of the tensor.
     */
    public static CNumber[] add(double[] src1, CNumber a) {
        CNumber[] sum = new CNumber[src1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = a.add(src1[i]);
        }

        return sum;
    }
}
