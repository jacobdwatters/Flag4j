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

package org.flag4j.operations_old.dense.complex;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.core.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This class contains low-level implementations of complex element-wise tensor multiplication.
 */
public class ComplexDenseElemDiv {

    /**
     * Minimum number of entries in each tensor to apply concurrent algorithm.
     */
    private static final int CONCURRENT_THRESHOLD = 1250;


    private ComplexDenseElemDiv() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the element-wise division of two tensors.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @return The element-wise division of the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CNumber[] elemDiv(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);
        CNumber[] product = new CNumber[src1.length];

        for(int i=0; i<product.length; i++) {
            product[i] = src1[i].div(src2[i]);
        }

        return product;
    }


    /**
     * Computes the element-wise division of two tensors using a concurrent algorithm.
     * @param src1 First tensor in element-wise division.
     * @param shape1 Shape of the first tensor.
     * @param src2 Second tensor in element-wise division.
     * @param shape2 Shape of the second tensor.
     * @return The element-wise division of the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CNumber[] elemDivConcurrent(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        ParameterChecks.assertEqualShape(shape1, shape2);
        CNumber[] product = new CNumber[src1.length];

        ThreadManager.concurrentOperation(product.length, (start, end)->{
            for(int i=start; i<end; i++) {
                product[i] = src1[i].div(src2[i]);
            }
        });

        return product;
    }


    /**
     * Dynamically chooses and applies element-wise division algorithm to use based on the number of entries in the tensors.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element-wise division of the two tensors.
     */
    public static CNumber[] dispatch(CNumber[] src1, Shape shape1,CNumber[] src2, Shape shape2) {
        if(src1.length < CONCURRENT_THRESHOLD) {
            return elemDiv(src1, shape1, src2, shape2);
        } else {
            return elemDivConcurrent(src1, shape1, src2, shape2);
        }
    }
}
