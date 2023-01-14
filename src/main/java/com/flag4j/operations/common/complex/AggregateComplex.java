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

package com.flag4j.operations.common.complex;


import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;

/**
 * This class contains several low-level methods useful for computing aggregation operations on dense/sparse complex tensors.
 */
public class AggregateComplex {

    private AggregateComplex() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the sum of all entries in this tensor. This can be applied to either real dense or spase tensors.
     * @param entries Entries of the tensor.
     * @return The sum of all entries in this tensor.
     */
    public static CNumber sum(final CNumber[] entries) {
        CNumber sum = new CNumber();
        for(CNumber value : entries) {
            sum.addEq(value);
        }
        return sum;
    }


    /**
     * Computes the minimum absolute value in a tensor. Note, if the entries array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The minimum absolute value in the tensor.
     */
    public static CNumber minAbs(final CNumber[] entries) {
        CNumber currMin;

        if(entries.length!=0) {
            currMin = new CNumber(Double.MAX_VALUE);

            for(CNumber value : entries) {
                if(value.mag().re < currMin.re) {
                    currMin = value.mag(); // Update current minimum.
                }
            }

        } else {
            currMin = new CNumber();
        }

        return currMin;
    }


    /**
     * Computes the maximum absolute value in a tensor. Note, if the entries array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The maximum absolute value in the tensor.
     */
    public static CNumber maxAbs(final CNumber[] entries) {
        CNumber currMax = new CNumber();

        for(CNumber value : entries) {
            if(value.mag().re > currMax.re) {
                currMax = value.mag(); // Update current maximum.
            }
        }

        return currMax;
    }
}
