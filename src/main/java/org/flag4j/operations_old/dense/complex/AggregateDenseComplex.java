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

package org.flag4j.operations_old.dense.complex;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ErrorMessages;

/**
 * This class contains several low-level methods useful for computing aggregation operations_old on dense tensors.
 */
public class AggregateDenseComplex {

    private AggregateDenseComplex() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the index of the minimum value by magnitude in this tensor. If the minimum value occurs at more than one index,
     * the index of the first occurrence is taken.
     * @param entries Entries of the tensor.
     * @return The index of the minimum value by magnitude in this tensor. If there are zero entries in the array, -1 is returned.
     */
    public static int argMin(CNumber[] entries) {
        double currMin = Double.MAX_VALUE;
        int currMinIndex = -1;

        for(int i=0; i<entries.length; i++) {
            if(entries[i].mag() < currMin) {
                currMin = entries[i].mag(); // Update current minimum.
                currMinIndex = i;
            }
        }

        return currMinIndex;
    }


    /**
     * Computes the index of the maximum value by magnitude in this tensor. If the maximum value occurs at more than one index,
     * the index of the first occurrence is taken.
     * @param entries Entries of the tensor.
     * @return The index of the maximum value by magnitude in this tensor. If there are zero entries in the array, -1 is returned.
     */
    public static int argMax(CNumber[] entries) {
        double currMax = Double.MIN_VALUE;
        int currMaxIndex = -1;

        for(int i=0; i<entries.length; i++) {
            if(entries[i].mag() > currMax) {
                currMax = entries[i].mag(); // Update current minimum.
                currMaxIndex = i;
            }
        }

        return currMaxIndex;
    }
}
