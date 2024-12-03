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

package org.flag4j.linalg.ops.dense;

import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.linalg.ops.TensorDot;
import org.flag4j.linalg.ops.TransposeDispatcher;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringMatMultDispatcher;

/**
 * Instances of this class can be used to compute the tensor dot product between two dense tensors.
 * @param <T> The type of semiring that the elements of the tensors in the dot product belong to.
 */
public class DenseSemiringTensorDot<T extends Semiring<T>> extends TensorDot<T[]> {

    /**
     * Constructs a tensor dot product problem for computing the tensor contraction of two tensors over the
     * specified set of axes. That is, computes the sum of products between the two tensors along the specified set of axes.
     * @param shape1 Shape of the first tensor in the contraction.
     * @param src1 Entries of the first tensor in the contraction.
     * @param shape2 Shape of the second tensor in the contraction.
     * @param src2 Entries of the second tensor in the contraction.
     * @param src1Axes Axes along which to compute products for {@code src1} tensor.
     * @param src2Axes Axes along which to compute products for {@code src2} tensor.
     * @throws IllegalArgumentException If {@code src1Axes} and {@code src2Axes} do not match in length, or if any of the axes
     * are out of bounds for the corresponding tensor. Or, If the two tensors shapes do not match along the specified axes pairwise
     * in {@code src1Axes} and {@code src2Axes}.
     */
    public DenseSemiringTensorDot(Shape shape1, T[] src1,
                                  Shape shape2, T[] src2,
                                  int[] src1Axes, int[] src2Axes) {
        super(shape1, src1, shape2, src2, src1Axes, src2Axes);
    }


    /**
     * Computes this tensor dot product as specified in the constructor.
     * @param dest The array to store the data of the tensor resulting from this tensor dot product. The size of this array
     * should be computed using {@link #getOutputSize()}.
     */
    @Override
    public void compute(T[] dest) {
        if(dest.length != destLength) {
            throw new IllegalArgumentException("dest array is not properly sized to store the result of the tensor dot product." +
                    " Expecting length " + destLength + " but got " + dest.length +  ". Try using calling" +
                    "getOutputSize() to determine the required size of the dest array.");
        }

        // Reform tensor dot product problem as a matrix multiplication problem.
        T[] at = (T[]) new Semiring[src1.length];
        T[] bt = (T[]) new Semiring[src2.length];
        TransposeDispatcher.dispatchTensor(src1, shape1, src1NewAxes, at);
        TransposeDispatcher.dispatchTensor(src2, shape2, src2NewAxes, bt);

        DenseSemiringMatMultDispatcher.dispatch(at, newShape1, bt, newShape2, dest);
    }
}
