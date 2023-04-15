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

package com.flag4j.operations.common.real;

import com.flag4j.util.ErrorMessages;

/**
 * This class provides low level methods for checking tensor properties. These methods can be applied to
 * either sparse or dense real tensors.
 */
public class RealProperties {

    private RealProperties() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if a tensor only contain positive values. If the tensor is sparse, only the non-zero entries are considered.
     * @param entries Entries of the tensor in question.
     * @return True if the tensor contains only positive values. Otherwise, returns false.
     */
    public static boolean isPos(double[] entries) {
        boolean result = true;

        for(double value : entries) {
            if(value<=0) {
                result = false;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if a tensor only contain negative values. If the tensor is sparse, only the non-zero entries are considered.
     * @param entries Entries of the tensor in question.
     * @return True if the tensor contains only negative values. Otherwise, returns false.
     */
    public static boolean isNeg(double[] entries) {
        boolean result = true;

        for(double value : entries) {
            if(value>=0) {
                result = false;
                break;
            }
        }

        return result;
    }
}
