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

package org.flag4j.operations.dense_sparse.coo.real;

import org.flag4j.dense.Tensor;
import org.flag4j.sparse.CooTensor;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This class contains methods to apply common binary operations to a real dense matrix and to a real sparse matrix.
 */
public class RealDenseSparseTensorOperations {

    private RealDenseSparseTensorOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real dense tensor to a real sparse tensor.
     * @param src1 First tensor in sum.
     * @param src2 Second tensor in sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static Tensor add(Tensor src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int[] indices;
        Tensor dest = new Tensor(src1);

        for(int i=0; i<src2.entries.length; i++) {
            indices = src2.indices[i];
            dest.entries[dest.shape.entriesIndex(indices)] += src2.entries[i];
        }

        return dest;
    }


    /**
     * Computes the element-wise multiplication between a real dense tensor and a real sparse tensor.
     * @param src1 Real dense tensor.
     * @param src2 Real sparse tensor.
     * @return The result ofm element-wise multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooTensor elemMult(Tensor src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int index;
        double[] destEntries = new double[src2.nonZeroEntries()];
        int[][] destIndices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy(src2.indices, destIndices);

        for(int i=0; i<destEntries.length; i++) {
            index = src2.shape.entriesIndex(src2.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src1.entries[index]*src2.entries[i];
        }

        return new CooTensor(src2.shape.copy(), destEntries, destIndices);
    }


    /**
     * Subtracts a real sparse tensor from a real dense tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static Tensor sub(Tensor src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        Tensor dest = new Tensor(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            dest.entries[dest.shape.entriesIndex(src2.indices[i])] -= src2.entries[i];
        }

        return dest;
    }


    /**
     * Adds a real dense tensor to a real sparse tensor and stores the result in the first tensor.
     * @param src1 First tensor in sum. Also, storage of result.
     * @param src2 Second tensor in sum.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static void addEq(Tensor src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src1.shape.entriesIndex(src2.indices[i])] += src2.entries[i];
        }
    }


    /**
     * Subtracts a real sparse tensor from a real dense tensor and stores the result in the dense tensor.
     * @param src1 First tensor in difference. Also, storage of result.
     * @param src2 Second tensor in difference.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static void subEq(Tensor src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src1.shape.entriesIndex(src2.indices[i])] -= src2.entries[i];
        }
    }
}
