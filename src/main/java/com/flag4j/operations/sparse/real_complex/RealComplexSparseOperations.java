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

import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.SparseVector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;


/**
 * This class contains low level implementations of operations on a real sparse tensor and a complex sparse tensor.
 */
public class RealComplexSparseOperations {

    private RealComplexSparseOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static CVector add(SparseVector src, CNumber a) {
        CNumber[] dest = new CNumber[src.size];
        ArrayUtils.fill(dest, a);

        for(int i=0; i<src.entries.length; i++) {
            dest[src.indices[i]].addEq(src.entries[i]);
        }

        return new CVector(dest);
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static CVector add(SparseCVector src, double a) {
        CNumber[] dest = new CNumber[src.size];
        ArrayUtils.fill(dest, a);

        for(int i=0; i<src.entries.length; i++) {
            dest[src.indices[i]].addEq(src.entries[i]);
        }

        return new CVector(dest);
    }
}
