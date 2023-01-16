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

import com.flag4j.CMatrix;
import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.concurrency.util.ErrorMessages;

import java.util.Arrays;

/**
 * This class provides methods for checking the equality of complex dense tensors.
 */
public class ComplexDenseEquals {

    private ComplexDenseEquals() {
        // Hide constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if two real dense matrices are equal.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean matrixEquals(CMatrix A, CMatrix B) {
        return tensorEquals(A.entries, A.shape, B.entries, B.shape);
    }


    /**
     * Checks if two dense tensors are equal.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return True if the two tensors are numerically element-wise equivalent.
     */
    private static boolean tensorEquals(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        return shape1.equals(shape2) && Arrays.equals(src1, src2);
    }
}
