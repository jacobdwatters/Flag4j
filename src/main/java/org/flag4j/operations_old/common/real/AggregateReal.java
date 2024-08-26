/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.operations_old.common.real;


import org.flag4j.util.ErrorMessages;

/**
 * This class contains several low-level methods useful for computing aggregation operations_old on dense/sparse tensors.
 */
public class AggregateReal {

    private AggregateReal() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the sum of all entries in this tensor. This can be applied to either real dense or spase tensors.
     * @param entries Entries of the tensor.
     * @return The sum of all entries in this tensor.
     */
    public static double sum(final double[] entries) {
        double sum = 0;
        for(double value : entries) {
            sum += value;
        }
        return sum;
    }


    /**
     * Computes the minimum value in a tensor. Note, if the entries array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The minimum value in the tensor.
     */
    public static double min(final double[] entries) {
        double currMin = (entries.length==0) ? 0 : Double.MAX_VALUE;

        for(double value : entries) {
            currMin = Math.min(value, currMin);
        }

        return currMin;
    }


    /**
     * Computes the maximum value in a tensor. Note, if the entries array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The maximum value in the tensor.
     */
    public static double max(final double[] entries) {
        double currMax = (entries.length==0) ? 0 : Double.MIN_NORMAL;

        for(double value : entries) {
            currMax = Math.max(value, currMax);
        }

        return currMax;
    }


    /**
     * Computes the minimum absolute value in a tensor. Note, if the entries array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The minimum absolute value in the tensor.
     */
    public static double minAbs(final double[] entries) {
        double currMin = (entries.length==0) ? 0 : Double.MAX_VALUE;

        for(double value : entries) {
            currMin = Math.min(Math.abs(value), currMin);
        }

        return currMin;
    }


    /**
     * Computes the maximum absolute value in a tensor. Note, if the entries array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The maximum absolute value in the tensor.
     */
    public static double maxAbs(final double... entries) {
        double currMax = 0;

        for(double value : entries) {
            currMax = Math.max(Math.abs(value), currMax);
        }

        return currMax;
    }
}
