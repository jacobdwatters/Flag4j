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
import com.flag4j.CTensor;
import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.MatrixMultiplyDispatcher;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class contains methods for computing a tensor dot product, i.e. tensor contraction, between two complex dense tensors.
 */
public class ComplexDenseTensorDot {

    private ComplexDenseTensorDot() {
        // Hide utility class constructor.
        throw new IllegalArgumentException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the tensor dot product along the first tensors last axis and the second tensors second-to-last axis.
     * @param src1 First tensor in the tensor product.
     * @param src2 Second tensor in the tensor product.
     * @return The tensor dot product along the first tensors last axis and the second tensors second-to-last axis.
     * @throws IllegalArgumentException If this tensors shape along the last axis does not match {@code src2} shape
     * along the second-to-last axis.
     */
    public static CTensor dot(CTensor src1, CTensor src2) {
        int src1Rank = src1.getRank();
        int src2Rank = src2.getRank();

        if(src1Rank==2 && src2Rank==2) {
            // Product is simply a matrix multiplication problem.
            return new CTensor(
                    new Shape(src1.shape.dims[0], src2.shape.dims[1]),
                    MatrixMultiplyDispatcher.dispatch(src1.entries, src1.shape, src2.entries, src2.shape)
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
    public static CTensor tensorDot(CTensor src1, CTensor src2, int[] src1Axes, int[] src2Axes) {
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
            n2 *= src1.shape.get(axis);
        }

        n1 = 1;
        int[] src1OldDims = new int[notin.length];
        pos = 0;
        for(int axis : notin) {
            n1 *= src1.shape.get(axis);
            src1OldDims[pos++] = src1.shape.get(axis);
        }

        Shape src1NewShape = new Shape(n1, n2);
        // -----------------------------------------------------

        // ---- Compute new axes and shapes for second tensor. ----
        notin = ArrayUtils.notinAxes(src2Axes, src2.getRank());
        int[] src2NewAxes = ArrayUtils.join(src2Axes, notin);

        n2 = 1;
        for(int axis : src2Axes) {
            n2 *= src2.shape.get(axis);
        }

        n1 = 1;
        pos = 0;
        int[] src2OldDims = new int[notin.length];
        for(int axis : notin) {
            n1 *= src2.shape.get(axis);
            src2OldDims[pos++] = src2.shape.get(axis);
        }

        Shape src2NewShape = new Shape(n2, n1);
        // -----------------------------------------------------

        // Reform problem as a matrix multiplication problem.
        CMatrix at = new CMatrix(
                src1NewShape,
                ComplexDenseTranspose.standardConcurrent(src1.entries, src1.shape, src1NewAxes)
        );
        CMatrix bt = new CMatrix(
                src2NewShape,
                ComplexDenseTranspose.standardConcurrent(src2.entries, src2.shape, src2NewAxes)
        );

        CNumber[] destEntries = MatrixMultiplyDispatcher.dispatch(at, bt);
        Shape destShape = new Shape(ArrayUtils.join(src1OldDims, src2OldDims));

        return new CTensor(destShape, destEntries);
    }
}
