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

package org.flag4j.linalg.operations.sparse.coo.ring_ops;

import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.SparseVectorData;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for computing operations on sparse COO {@link org.flag4j.algebraic_structures.rings.Ring} vectors.
 */
public final class CooRingVectorOps {


    private CooRingVectorOps() {
        // Hide default constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(getClass()));
    }


    /**
     * Computes the element-wise vector subtraction between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the difference. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the difference. Indices assumed to be sorted lexicographically.
     * @return A data class containing the non-zero entries and indices of the sparse COO vector resulting from the vector subtraction.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends Ring<T>> SparseVectorData<Ring<T>> sub(
            Shape shape1, Ring<T>[] src1, int[] src1Indices,
            Shape shape2, Ring<T>[] src2, int[] src2Indices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        List<Ring<T>> values = new ArrayList<>(src1.length);
        List<Integer> indices = new ArrayList<>(src1.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.length && src2Counter < src2.length) {
            if(src1Indices[src1Counter] == src2Indices[src2Counter]) {
                values.add(src1[src1Counter].sub((T) src2[src2Counter]));
                indices.add(src1Indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1Indices[src1Counter] < src2Indices[src2Counter]) {
                values.add(src1[src1Counter]);
                indices.add(src1Indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2[src2Counter].addInv());
                indices.add(src2Indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.length) {
            for(int i=src1Counter; i<src1.length; i++) {
                values.add(src1[i]);
                indices.add(src1Indices[i]);
            }
        } else if(src2Counter < src2.length) {
            for(int i=src2Counter; i<src2.length; i++) {
                values.add(src2[i].addInv());
                indices.add(src2Indices[i]);
            }
        }

        return new SparseVectorData<Ring<T>>(shape1, values, indices);
    }
}
