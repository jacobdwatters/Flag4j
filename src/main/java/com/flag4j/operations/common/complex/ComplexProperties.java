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
 * This class contains low-level implementations for operations which check if a complex tensor satisfies some property.
 * Implementations are agnostic to whether the tensor is sparse or dense.
 */
public final class ComplexProperties {

    private ComplexProperties() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks whether a tensor contains only real values.
     * @param entries Entries of dense tensor or non-zero entries of sparse tensor.
     * @return True if the tensor only contains real values. Returns false otherwise.
     */
    public static boolean isReal(CNumber[] entries) {
        boolean result = true;

        for(int i=0; i<entries.length; i++) {
            if(entries[i].im != 0) {
                result = false;
                break;
            }
        }

        return result;
    }


    /**
     * Checks whether a tensor contains at least one non-real value.
     * @param entries Entries of dense tensor or non-zero entries of sparse tensor.
     * @return True if the tensor contains at least one non-real value. Returns false otherwise.
     */
    public static boolean isComplex(CNumber[] entries) {
        boolean result = false;

        for(int i=0; i<entries.length; i++) {
            if(entries[i].im != 0) {
                result = true;
                break;
            }
        }

        return result;
    }
}
