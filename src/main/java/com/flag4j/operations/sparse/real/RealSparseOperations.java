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

package com.flag4j.operations.sparse.real;


import com.flag4j.SparseVector;
import com.flag4j.Vector;
import com.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * This class contains low level implementations of operations on two real sparse tensors.
 */
public class RealSparseOperations {

    private RealSparseOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static Vector add(SparseVector src, double a) {
        double[] dest = new double[src.size];
        Arrays.fill(dest, a);

        for(int i=0; i<src.entries.length; i++) {
            dest[src.indices[i]] += src.entries[i];
        }

        return new Vector(dest);
    }
}
