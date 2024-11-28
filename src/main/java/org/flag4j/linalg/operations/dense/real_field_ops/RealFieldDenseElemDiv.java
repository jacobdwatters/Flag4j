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

package org.flag4j.linalg.operations.dense.real_field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains low-level implementations of element-wise tensor division for a dense real and dense
 * field tensor.
 */
public final class RealFieldDenseElemDiv {

    // TODO: This should be configurable
    /**
     * Minimum number of data in each tensor to apply concurrent algorithm.
     */
    private static final int CONCURRENT_THRESHOLD = 15_625;

    private RealFieldDenseElemDiv() {
        // Hide default constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
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
    public static <T extends Field<T>> Field<T>[] elemDiv(Field<T>[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Field<T>[] product = new Field[src1.length];

        for(int i=0; i<product.length; i++)
            product[i] = src1[i].div(src2[i]);

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
    public static <T extends Field<T>> Field<T>[] elemDivConcurrent(Field<T>[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Field<T>[] product = new Field[src1.length];

        ThreadManager.concurrentOperation(product.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++)
                product[i] = src1[i].div(src2[i]);
        });

        return product;
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
    public static <T extends Field<T>> Field<T>[] elemDiv(double[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Field<T>[] quotient = new Field[src1.length];

        for(int i=0, size=quotient.length; i<size; i++)
            quotient[i] = src2[i].multInv().mult(src1[i]);

        return quotient;
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
    public static <T extends Field<T>> Field<T>[] elemDivConcurrent(double[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        Field<T>[] quotient = new Field[src1.length];

        ThreadManager.concurrentOperation(quotient.length, (startIdx, endIdx) -> {
            for(int i=0, size=quotient.length; i<size; i++)
                quotient[i] = src2[i].multInv().mult(src1[i]);
        });

        return quotient;
    }


    /**
     * Chooses if a concurrent algorithm for element-wise multiplication should be used based on the shape of the two tensors.
     * @param numEntries Total data in the tensors to multiply.
     * @return True if a concurrent algorithm should be used for element-wise multiplication. Otherwise, returns false.
     */
    private static boolean useConcurrent(int numEntries) {
        return numEntries >= CONCURRENT_THRESHOLD;
    }


    /**
     * Dynamically chooses and applies the appropriate algorithm for element-wise tensor multiplication.
     *
     * @param shape1 Shape of first tensor.
     * @param src1 Entries of first tensor.
     * @param shape2 Shape of second tensor.
     * @param src2 Entries of second tensor.
     *
     * @return The result of the element-wise tensor multiplication.
     */
    public static <T extends Field<T>> Field<T>[] dispatch(Shape shape1, double[] src1, Shape shape2, Field<T>[] src2) {
        if(useConcurrent(src1.length)) {
            // Use concurrent algorithm.
            return elemDivConcurrent(src1, shape1, src2, shape2);
        } else {
            // Then use standard algorithm
            return elemDiv(src1, shape1, src2, shape2);
        }
    }


    /**
     * Dynamically chooses and applies the appropriate algorithm for element-wise tensor multiplication.
     * @param src1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param src2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The result of the element-wise tensor multiplication.
     */
    public static <T extends Field<T>> Field<T>[] dispatch(Field<T>[] src1, Shape shape1, double[] src2, Shape shape2) {
        if(useConcurrent(src1.length)) {
            // Use concurrent algorithm.
            return elemDivConcurrent(src1, shape1, src2, shape2);
        } else {
            // Then use standard algorithm
            return elemDiv(src1, shape1, src2, shape2);
        }
    }
}
