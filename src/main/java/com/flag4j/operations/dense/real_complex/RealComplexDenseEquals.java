/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j.operations.dense.real_complex;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;

/**
 * This class provides methods for checking the equality of one real and one complex dense tensors.
 */
public class RealComplexDenseEquals {

    private RealComplexDenseEquals() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Checks if two real dense matrices are equal.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean matrixEquals(Matrix A, CMatrix B) {
        return A.shape.equals(B.shape) && ArrayUtils.equals(A.entries, B.entries);
    }
}
