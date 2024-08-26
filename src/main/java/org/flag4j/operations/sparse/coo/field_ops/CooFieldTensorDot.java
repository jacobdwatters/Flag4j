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

package org.flag4j.operations.sparse.coo.field_ops;

import org.flag4j.core.Shape;
import org.flag4j.core_temp.arrays.dense.DenseFieldTensorBase;
import org.flag4j.core_temp.arrays.dense.FieldTensor;
import org.flag4j.core_temp.arrays.sparse.CooFieldTensorBase;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.operations.dense.real.RealDenseTranspose;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;


/**
 * <p>Utility class for computing tensor dot products between two COO field tensors.</p>
 */
public final class CooFieldTensorDot {

    private CooFieldTensorDot() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
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
    public static <T extends CooFieldTensorBase<T, U, V>, U
            extends DenseFieldTensorBase<U, T, V>, V
            extends Field<V>> FieldTensor<V> tensorDot(
                    CooFieldTensorBase<T, U, V> src1, CooFieldTensorBase<T, U, V> src2,
                    int[] src1Axes, int[] src2Axes) {
        // Each array must specify the same number of axes.
        ParameterChecks.ensureEquals(src1Axes.length, src2Axes.length);

        // Axis values must be less than the rank of the tensor and non-negative
        ParameterChecks.ensureLessEq(src1.getRank()-1, src1Axes);
        ParameterChecks.ensureGreaterEq(0, src1Axes);
        ParameterChecks.ensureLessEq(src2.getRank()-1, src2Axes);
        ParameterChecks.ensureGreaterEq(0, src2Axes);

        int[] notin;
        int n1;
        int n2;
        int pos;

        // ---- Compute new axes and shapes for first tensor. ----
        notin = ArrayUtils.notInAxes(src1Axes, src1.getRank());
        int[] src1NewAxes = ArrayUtils.join(notin, src1Axes);

        n2 = 1;
        for(int axis : src1Axes) {
            n2 *= src1.shape.get(axis);
        }

        n1 = 1;
        int[] src1OldDims = new int[notin.length];
        pos = 0;
        for(int axis : notin) {
            int a = src1.shape.get(axis);
            n1 *= a;
            src1OldDims[pos++] = a;
        }

        Shape src1NewShape = new Shape(n1, n2);
        // -----------------------------------------------------

        // ---- Compute new axes and shapes for second tensor. ----
        notin = ArrayUtils.notInAxes(src2Axes, src2.getRank());
        int[] src2NewAxes = ArrayUtils.join(src2Axes, notin);

        n2 = 1;
        for(int axis : src2Axes) {
            n2 *= src2.shape.get(axis);
        }

        n1 = 1;
        pos = 0;
        int[] src2OldDims = new int[notin.length];
        for(int axis : notin) {
            int a = src2.shape.get(axis);
            n1 *= a;
            src2OldDims[pos++] = a;
        }

        Shape src2NewShape = new Shape(n2, n1);
        // -----------------------------------------------------

        // Reform tensor dot product problem as a matrix multiplication problem.
        T at = src1.T(src1NewAxes).reshape(src1NewShape);
        T bt = src2.T(src2NewAxes).reshape(src2NewShape);

        int[][] atMatIndices = RealDenseTranspose.blockedIntMatrix(at.indices);
        int[][] btMatIndices = RealDenseTranspose.blockedIntMatrix(bt.indices);

        Field<V>[] productEntries = CooFieldMatMult.standard(
                at.entries, atMatIndices[0], atMatIndices[1], at.shape,
                bt.entries, btMatIndices[0], btMatIndices[1], bt.shape
        );

        // TODO: Should allow for zero dim shape indicating a scalar. Then only the else block would be needed.
        Shape productShape = new Shape(ArrayUtils.join(src1OldDims, src2OldDims));

        return new FieldTensor<V>(productShape, (V[]) productEntries);
    }
}