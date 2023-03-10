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

package com.flag4j.operations.dense.complex;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;


/**
 * This class provides low level implementations for vector operations with two complex dense vectors.
 */
public class ComplexDenseVectorOperations {

    private ComplexDenseVectorOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the vector inner product for two complex vectors.
     * @param src1 Entries of the first vector.
     * @param src2 Entries of the second vector.
     * @return The inner product of the two vectors.
     */
    public static CNumber innerProduct(CNumber[] src1, CNumber[] src2) {
        ParameterChecks.assertArrayLengthsEq(src1.length, src2.length);
        CNumber innerProd = new CNumber();

        for(int i=0; i<src1.length; i++) {
            innerProd.addEq(src1[i].mult(src2[i].conj()));
        }

        return innerProd;
    }



    /**
     * Computes the vector outer product between two real dense vectors.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static CNumber[] outerProduct(CNumber[] src1, CNumber[] src2) {
        int destIndex;
        CNumber[] dest = new CNumber[src1.length*src2.length];

        for(int i=0; i<src1.length; i++) {
            destIndex = i*src2.length;
            for(int j=0; j<src2.length; j++) {
                dest[destIndex++] = src1[i].mult(src2[j].conj());
            }
        }

        return dest;
    }



}
