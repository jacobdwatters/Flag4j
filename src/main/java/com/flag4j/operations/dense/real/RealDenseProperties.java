/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

package com.flag4j.operations.dense.real;

import com.flag4j.Shape;
import com.flag4j.util.ErrorMessages;

/**
 * This class contains low-level implementations for operations which check if a tensor satisfies some property.
 */
public class RealDenseProperties {

    private RealDenseProperties() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if this tensor only contains ones.
     * @param src Elements of the tensor.
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    public static boolean isOnes(double[] src) {
        boolean allZeros = true;

        for(double value : src) {
            if(value != 1) {
                allZeros = false;
                break; // No need to look further.
            }
        }

        return allZeros;
    }


    /**
     * Checks if a real dense matrix is symmetric. That is, if the and equal to its transpose.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return True if this matrix is symmetric
     */
    public static boolean isSymmetric(double[] src, Shape shape) {
        if(shape.dims[0]!=shape.dims[1]) {
            return false;
        }

        int count1, count2, stop;

        for(int i=0; i<shape.dims[0]; i++) {
            count1 = i*shape.dims[1];
            count2 = i;
            stop = count1 + i;

            while(count1 < stop) {
                if(src[count1++] != src[count2]) {
                    return false;
                }

                count2+=shape.dims[1];
            }
        }

        return true;
    }


    /**
     * Checks if a real dense matrix is anti-symmetric. That is, if the and equal to its negative transpose.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return True if this matrix is anti-symmetric
     */
    public static boolean isAntiSymmetric(double[] src, Shape shape) {
        if(shape.dims[0]!=shape.dims[1]) {
            return false;
        }

        int count1, count2, stop;

        for(int i=0; i<shape.dims[0]; i++) {
            count1 = i*shape.dims[1];
            count2 = i;
            stop = count1 + i;

            while(count1 < stop) {
                if(src[count1++] != -src[count2]) {
                    return false;
                }

                count2+=shape.dims[1];
            }
        }

        return true;
    }
}
