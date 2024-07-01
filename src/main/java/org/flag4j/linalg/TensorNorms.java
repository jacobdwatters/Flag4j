/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.linalg;

import org.flag4j.core.dense_base.ComplexDenseTensorBase;
import org.flag4j.core.dense_base.RealDenseTensorBase;
import org.flag4j.util.ErrorMessages;


/**
 * Utility class for computing norms of tensors.
 */
public class TensorNorms {

    private TensorNorms() {
        // Hide default constructor for utility class
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest absolute value.
     * @param src The tensor, matrix, or vector to compute the norm of.
     * @return The infinity norm of the source matrix.
     */
    public static double infNorm(RealDenseTensorBase<?, ?> src) {
        return src.maxAbs();
    }


    /**
     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest value by magnitude.
     * @param src The tensor, matrix, or vector to compute the norm of.
     * @return The infinity norm of the source matrix.
     */
    public static double infNorm(ComplexDenseTensorBase<?, ?> src) {
        return src.maxAbs();
    }
}
