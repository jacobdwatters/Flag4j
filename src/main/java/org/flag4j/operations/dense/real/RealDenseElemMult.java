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

package org.flag4j.operations.dense.real;


import org.flag4j.concurrency.ThreadManager;
import org.flag4j.core.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This class contains low level implementations of element-wise multiplications algorithms for real dense tensors.
 */
public class RealDenseElemMult {

    /**
     * Minimum number of entries in each tensor to apply concurrent algorithm.
     */
    private static final int CONCURRENT_THRESHOLD = 30_000_000;

    private RealDenseElemMult() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the element-wise multiplication of two tensors. Also called the Hadamard product.
     * @param src1 First tensor in element-wise multiplication.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise multiplication.
     * @param shape2 Shape of the second tensor.
     * @return The element-wise multiplication of the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static double[] elemMult(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);
        double[] product = new double[src1.length];

        for(int i=0; i<product.length; i++) {
            product[i] = src1[i]*src2[i];
        }

        return product;
    }


    /**
     * Computes the element-wise multiplication of two tensors using a concurrent algorithm. Also called the Hadamard product.
     * @param src1 First tensor in element-wise multiplication.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise multiplication.
     * @param shape2 Shape of the second tensor.
     * @return The element-wise multiplication of the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static double[] elemMultConcurrent(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);
        double[] product = new double[src1.length];

        ThreadManager.concurrentOperation(product.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                product[i] = src1[i]*src2[i];
            }
        });

        return product;
    }


    /**
     * Dynamically chooses and applies element-wise multiplication algorithm to use based on the number of entries in the tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element-wise multiplication of the two tensors.
     */
    public static double[] dispatch(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        if(src1.length < CONCURRENT_THRESHOLD) {
            return elemMult(src1, shape1, src2, shape2);
        } else {
            return elemMultConcurrent(src1, shape1, src2, shape2);
        }
    }
}
