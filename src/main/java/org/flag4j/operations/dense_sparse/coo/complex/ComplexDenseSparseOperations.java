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

package org.flag4j.operations.dense_sparse.coo.complex;


import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This class contains methods to apply common binary operations to a complex dense/sparse matrix and to a complex sparse/dense matrix.
 */
public class ComplexDenseSparseOperations {

    private ComplexDenseSparseOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes element-wise addition of a complex dense tensor with a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise subtraction.
     */
    public static CTensor add(CTensor src1, CooCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CTensor dest = new CTensor(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            dest.entries[src2.shape.entriesIndex(src2.indices[i])].addEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Computes element-wise addition of a complex dense tensor with a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise addition.
     */
    public static void addEq(CTensor src1, CooCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src2.shape.entriesIndex(src2.indices[i])].addEq(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise tensor a complex sparse tensor from a complex dense tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise tensor subtraction.
     */
    public static CTensor sub(CTensor src1, CooCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CTensor dest = new CTensor(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            dest.entries[src2.shape.entriesIndex(src2.indices[i])].subEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a complex dense tensor from a complex sparse tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static CTensor sub(CooCTensor src1, CTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CTensor dest = src2.mult(-1);

        for(int i=0; i<src1.nnz; i++) {
            dest.entries[src1.shape.entriesIndex(src1.indices[i])].addEq(src1.entries[i]);
        }

        return dest;
    }


    /**
     * Computes element-wise subtraction of a complex dense tensor with a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of the element-wise subtraction.
     */
    public static void subEq(CTensor src1, CooCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src2.shape.entriesIndex(src2.indices[i])].subEq(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise tensor multiplication between a complex dense tensor and a complex sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Complex sparse tensor.
     * @return THe result of the element-wise tensor multiplication.
     */
    public static CooCTensor elemMult(CTensor src1, CooCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] destEntries = new CNumber[src2.nonZeroEntries()];
        int[][] indices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy(src2.indices, indices);

        for(int i=0; i<destEntries.length; i++) {
            destEntries[i] = src1.entries[src2.shape.entriesIndex(src2.indices[i])].mult(src2.entries[i]);
        }

        return new CooCTensor(src2.shape, destEntries, indices);
    }


    /**
     * Adds a scalar to a complex sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static CTensor add(CooCTensor src1, CNumber b) {
        CTensor sum = new CTensor(src1.shape, b);

        for(int i=0; i<src1.nnz; i++) {
            sum.entries[src1.shape.entriesIndex(src1.indices[i])].addEq(src1.entries[i]);
        }

        return sum;
    }
}
