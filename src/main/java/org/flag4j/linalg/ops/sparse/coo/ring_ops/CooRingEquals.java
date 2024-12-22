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

package org.flag4j.linalg.ops.sparse.coo.ring_ops;


import org.flag4j.algebraic_structures.Ring;
import org.flag4j.arrays.backend.ring_arrays.AbstractCooRingMatrix;
import org.flag4j.arrays.backend.ring_arrays.AbstractCooRingTensor;
import org.flag4j.arrays.backend.ring_arrays.AbstractCooRingVector;

import java.util.Arrays;

/**
 * A utility class for checking equality between COO {@link Ring} tensors.
 */
public class CooRingEquals {

    private CooRingEquals() {
        // Hide default constructor.
    }

    /**
     * Checks if two real sparse tensors are real. Assumes the indices of each sparse tensor are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First tensor in the equality check.
     * @param b Second tensor in the equality check.
     * @return True if the tensors are equal. False otherwise.
     */
    public static <T extends Ring<T>> boolean cooTensorEquals(
            AbstractCooRingTensor<?, ?, T> a,
            AbstractCooRingTensor<?, ?, T> b) {
        if (a == b) return true;
        if (a == null || b == null) return false;

        a = a.coalesce().dropZeros();
        b = b.coalesce().dropZeros();
        return a.shape.equals(b.shape)
                && Arrays.equals(a.data, b.data)
                && Arrays.deepEquals(a.indices, b.indices);
    }


    /**
     * Checks if two real sparse matrices are real. Assumes the indices of each sparse matrix are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First matrix in the equality check.
     * @param b Second matrix in the equality check.
     * @return True if the matrices are equal. False otherwise.
     */
    public static <T extends Ring<T>> boolean cooMatrixEquals(
            AbstractCooRingMatrix<?, ?, ?, T> a,
            AbstractCooRingMatrix<?, ?, ?, T> b) {
        // Early return if possible.
        if (a == b) return true;
        if (a == null || b == null) return false;

        a = a.coalesce().dropZeros();
        b = b.coalesce().dropZeros();
        return a.shape.equals(b.shape)
                && Arrays.equals(a.data, b.data)
                && Arrays.equals(a.rowIndices, b.rowIndices)
                && Arrays.equals(a.colIndices, b.colIndices);
    }


    /**
     * Checks if two real sparse vectors are real. Assumes the indices of each sparse vector are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static <T extends Ring<T>> boolean cooVectorEquals(
            AbstractCooRingVector<?, ?, ?, ?, T> a,
            AbstractCooRingVector<?, ?, ?, ?, T> b) {
        // Early returns if possible.
        if(a == b) return true;
        if(a==null || b==null || !a.shape.equals(b.shape)) return false;

        a = a.coalesce().dropZeros();
        b = b.coalesce().dropZeros();
        return a.shape.equals(b.shape)
                && Arrays.equals(a.data, b.data)
                && Arrays.equals(a.indices, b.indices);
    }
}
