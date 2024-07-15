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

import org.flag4j.arrays.dense.Tensor;
import org.flag4j.core.Shape;
import org.flag4j.operations.RealDenseMatrixMultiplyDispatcher;
import org.flag4j.operations.TransposeDispatcher;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;


/**
 * This class contains methods for computing a tensor dot product, i.e. tensor contraction, between two real dense tensors.
 */
public final class RealDenseTensorDot {

    private RealDenseTensorDot() {
        // Hide utility class constructor.
        throw new IllegalArgumentException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the tensor dot product along the first tensors last axis and the second tensors second-to-last axis. Or, if the second
     * tensor has rank 1, the last axis of the second tensor.
     * @param src1 First tensor in the tensor product.
     * @param src2 Second tensor in the tensor product.
     * @return The tensor dot product along the first tensors last axis and the second tensors second-to-last axis.
     * @throws IllegalArgumentException If this tensors shape along the last axis does not match {@code src2} shape
     * along the second-to-last axis.
     */
    public static Tensor tensorDot(Tensor src1, Tensor src2) {
        int src1Rank = src1.getRank();
        int src2Rank = src2.getRank();

        if(src1Rank==2 && src2Rank==2) {
            // Product is simply a matrix multiplication problem.
            return new Tensor(
                    new Shape(src1.shape.dims[0], src2.shape.dims[1]),
                    RealDenseMatrixMultiplyDispatcher.dispatch(src1.entries, src1.shape, src2.entries, src2.shape)
            );
        }

        // If second tensor has rank one, then use zero axis. Otherwise, use second to last axis.
        src2Rank = (src2Rank==1) ? 0 : src2Rank-2;

        return tensorDot(src1, src2, new int[]{src1Rank - 1}, new int[]{src2Rank});
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     * @param src1 First tensor in the contraction.
     * @param src2 Second tensor in the contraction.
     * @param src1Axes Axes along which to compute products for {@code src1} tensor.
     * @param src2Axes Axes along which to compute products for {@code src2} tensor.
     * @return The tensor dot product over the specified axes.
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     * {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     * are out of bounds for the corresponding tensor.
     */
    public static Tensor tensorDot(Tensor src1, Tensor src2, int[] src1Axes, int[] src2Axes) {
        if(src1.getRank()==2 && src2.getRank()==2) {

        }


        // Each array must specify the same number of axes.
        ParameterChecks.assertEquals(src1Axes.length, src2Axes.length);

        // Axis values must be less than the rank of the tensor and non-negative
        ParameterChecks.assertLessEq(src1.getRank()-1, src1Axes);
        ParameterChecks.assertGreaterEq(0, src1Axes);
        ParameterChecks.assertLessEq(src2.getRank()-1, src2Axes);
        ParameterChecks.assertGreaterEq(0, src2Axes);

        int[] notin;
        int n1;
        int n2;
        int pos;

        // ---- Compute new axes and shapes for first tensor. ----
        notin = ArrayUtils.notinAxes(src1Axes, src1.getRank());
        int[] src1NewAxes = ArrayUtils.join(notin, src1Axes);

        n2 = 1;
        for(int axis : src1Axes) {
            n2 *= src1.shape.dims[axis];
        }

        n1 = 1;
        int[] src1OldDims = new int[notin.length];
        pos = 0;
        for(int axis : notin) {
            int a = src1.shape.dims[axis];
            n1 *= a;
            src1OldDims[pos++] = a;
        }

        Shape src1NewShape = new Shape(n1, n2);
        // -----------------------------------------------------

        // ---- Compute new axes and shapes for second tensor. ----
        notin = ArrayUtils.notinAxes(src2Axes, src2.getRank());
        int[] src2NewAxes = ArrayUtils.join(src2Axes, notin);

        n2 = 1;
        for(int axis : src2Axes) {
            n2 *= src2.shape.dims[axis];
        }

        n1 = 1;
        pos = 0;
        int[] src2OldDims = new int[notin.length];
        for(int axis : notin) {
            int a = src2.shape.dims[axis];
            n1 *= a;
            src2OldDims[pos++] = a;
        }

        Shape src2NewShape = new Shape(n2, n1);
        // -----------------------------------------------------

        // Reform tensor dot product problem as a matrix multiplication problem.
        double[] at = TransposeDispatcher.dispatchTensor(src1.entries, src1.shape, src1NewAxes);
        double[] bt = RealDenseTranspose.standardConcurrent(src2.entries, src2.shape, src2NewAxes);

        double[] destEntries = RealDenseMatrixMultiplyDispatcher.dispatch(at, src1NewShape, bt, src2NewShape);
        Shape destShape = new Shape(ArrayUtils.join(src1OldDims, src2OldDims));

        return new Tensor(destShape, destEntries);
    }
}
