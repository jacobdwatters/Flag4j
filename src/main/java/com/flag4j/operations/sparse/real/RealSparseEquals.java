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

package com.flag4j.operations.sparse.real;


import com.flag4j.SparseMatrix;
import com.flag4j.SparseTensor;
import com.flag4j.SparseVector;

import java.util.Arrays;

/**
 * This class contains methods for checking the equality of real sparse tensors.
 */
public class RealSparseEquals {


    /**
     * Checks if two real sparse tensors are real. Assumes the indices of each sparse tensor are sorted.
     * @param a First tensor in the equality check.
     * @param b Second tensor in the equality check.
     * @return True if the tensors are equal. False otherwise.
     */
    public static boolean tensorEquals(SparseTensor a, SparseTensor b) {
        return a.shape.equals(b.shape) && Arrays.equals(a.entries, b.entries)
                && Arrays.deepEquals(a.indices, b.indices);
    }


    /**
     * Checks if two real sparse matrices are real. Assumes the indices of each sparse matrix are sorted.
     * @param a First matrix in the equality check.
     * @param b Second matrix in the equality check.
     * @return True if the matrices are equal. False otherwise.
     */
    public static boolean matrixEquals(SparseMatrix a, SparseMatrix b) {
        return a.shape.equals(b.shape) && Arrays.equals(a.entries, b.entries)
                && Arrays.equals(a.rowIndices, b.rowIndices) && Arrays.equals(a.colIndices, b.colIndices);
    }


    /**
     * Checks if two real sparse vectors are real. Assumes the indices of each sparse vector are sorted.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static boolean vectorEquals(SparseVector a, SparseVector b) {
        return a.size == b.size && Arrays.equals(a.indices, b.indices)
                && Arrays.equals(a.entries, b.entries);
    }
}
