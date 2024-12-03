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

package org.flag4j.linalg.ops.dense.ring_ops;

import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

/**
 * Utility class for computing ops between two dense {@link org.flag4j.algebraic_structures.rings.Ring} tensors.
 */
public final class DenseRingTensorOps {

    private DenseRingTensorOps() {
        // Hide default constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(getClass()));
    }


    /**
     * Computes the element-wise difference between two dense tensors.
     * @param shape1 Shape of the first tensor in the element-wise difference.
     * @param src1 Entries of the first tensor in the element-wise difference.
     * @param shape2 Shape of the second tensor in the element-wise difference.
     * @param src2 Entries of the second tensor in the element-wise difference.
     * @param dest Array to store the resulting element-wise difference. May be the same array as either {@code src1} or {@code src2}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !shape1.equals(shape2)}.
     * @throws ArrayIndexOutOfBoundsException If {@code src2.length < src2.length || dest.length < src1.length}
     */
    public static <T extends Ring<T>> void sub(Shape shape1, T[] src1,
                                               Shape shape2, T[] src2,
                                               T[] dest) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        for(int i=0, size=src1.length; i<size; i++)
            dest[i] = src1[i].sub(src2[i]);
    }
}
