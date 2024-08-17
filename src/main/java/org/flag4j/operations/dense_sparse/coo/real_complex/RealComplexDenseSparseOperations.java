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

package org.flag4j.operations.dense_sparse.coo.real_complex;

import org.flag4j.arrays_old.dense.CTensorOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.arrays_old.sparse.CooCTensor;
import org.flag4j.arrays_old.sparse.CooTensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This class contains methods to apply common binary operations_old to a real/complex dense matrix and to a complex/real sparse matrix.
 */
public final class RealComplexDenseSparseOperations {

    private RealComplexDenseSparseOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the element-wise multiplication between a real dense tensor and a complex sparse tensor.
     * @param src1 Real dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of element-wise multiplication between the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooCTensor elemMult(TensorOld src1, CooCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int index;
        CNumber[] destEntries = new CNumber[src2.nonZeroEntries()];
        int[][] destIndices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy(src2.indices, destIndices);

        for(int i=0; i<destEntries.length; i++) {
            index = src2.shape.entriesIndex(src2.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src2.entries[i].mult(src1.entries[index]);
        }

        return new CooCTensor(src2.shape, destEntries, destIndices);
    }


    /**
     * Computes the element-wise division between a real dense tensor and a complex sparse tensor.
     * @param src1 Real sparse tensor.
     * @param src2 complex dense tensor.
     * @return The result of element-wise division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooCTensor elemDiv(CooTensor src1, CTensorOld src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int index;
        CNumber[] destEntries = new CNumber[src1.nonZeroEntries()];
        int[][] destIndices = new int[src1.indices.length][src1.indices[0].length];
        ArrayUtils.deepCopy(src1.indices, destIndices);

        for(int i=0; i<destEntries.length; i++) {
            index = src2.shape.entriesIndex(src1.indices[i]); // Get index of non-zero entry.
            destEntries[i] = new CNumber(src1.entries[index]).div(src2.entries[i]);
        }

        return new CooCTensor(src2.shape, destEntries, destIndices);
    }


    /**
     * Computes the element-wise division between a real dense tensor and a real sparse tensor.
     * @param src1 Real sparse tensor.
     * @param src2 Real dense tensor.
     * @return The result of element-wise division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooCTensor elemDiv(CooCTensor src1, TensorOld src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int index;
        CNumber[] destEntries = new CNumber[src1.nonZeroEntries()];
        int[][] destIndices = new int[src1.indices.length][src1.indices[0].length];
        ArrayUtils.deepCopy(src1.indices, destIndices);

        for(int i=0; i<destEntries.length; i++) {
            index = src2.shape.entriesIndex(src1.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src1.entries[index].div(src2.entries[i]);
        }

        return new CooCTensor(src2.shape, destEntries, destIndices);
    }


    /**
     * Adds a real dense tensor to a sparse complex tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static CTensorOld add(TensorOld src1, CooCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CTensorOld dest = new CTensorOld(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = dest.shape.entriesIndex(src2.indices[i]);
            dest.entries[idx] = dest.entries[idx].add(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a sparse complex tensor from a real dense tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static CTensorOld sub(TensorOld src1, CooCTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CTensorOld dest = new CTensorOld(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = dest.shape.entriesIndex(src2.indices[i]);
            dest.entries[idx] = dest.entries[idx].sub(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Adds a dense complex tensor to a real sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Real sparse tensor.
     * @return The result of the tensor addition.
     */
    public static CTensorOld add(CTensorOld src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CTensorOld dest = new CTensorOld(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = dest.shape.entriesIndex(src2.indices[i]);
            dest.entries[idx] = dest.entries[idx].add(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Adds a dense complex tensor to a real sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Real sparse tensor.
     * @return The result of the tensor addition.
     */
    public static CTensorOld sub(CTensorOld src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CTensorOld dest = new CTensorOld(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = dest.shape.entriesIndex(src2.indices[i]);
            dest.entries[idx] = dest.entries[idx].sub(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Computes element-wise addition between a complex dense tensor and a real sparse tensor. The result is stored
     * in the complex dense tensor.
     * @param src1 The complex dense tensor. Also, the storage for the computation.
     * @param src2 The real sparse tensor.
     */
    public static void addEq(CTensorOld src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = src2.shape.entriesIndex(src2.indices[i]);
            src1.entries[idx] = src1.entries[idx].add(src2.entries[i]);
        }
    }


    /**
     * Computes element-wise subtraction between a complex dense tensor and a real sparse tensor. The result is stored
     * in the complex dense tensor.
     * @param src1 The complex dense tensor. Also, the storage for the computation.
     * @param src2 The real sparse tensor.
     */
    public static void subEq(CTensorOld src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            int idx = src2.shape.entriesIndex(src2.indices[i]);
            src1.entries[idx] = src1.entries[idx].sub(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise multiplication between a complex dense tensor and a real sparse matrix.
     * @param src1 First tensor in the element-wise multiplication.
     * @param src2 Second tensor in the element-wise multiplication.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static CooCTensor elemMult(CTensorOld src1, CooTensor src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] destEntries = new CNumber[src2.nonZeroEntries()];

        int[][] indices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy( src2.indices, indices);

        for(int i=0; i<destEntries.length; i++) {
            destEntries[i] = src1.entries[src2.shape.entriesIndex(src2.indices[i])].mult(src2.entries[i]);
        }

        return new CooCTensor(src2.shape, destEntries, indices);
    }


    /**
     * Subtracts a complex dense tensor from a real sparse tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static CTensorOld sub(CooTensor src1, CTensorOld src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CTensorOld dest = src2.mult(-1);

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.shape.entriesIndex(src1.indices[i]);
            dest.entries[idx] = dest.entries[idx].add(src1.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real dense tensor from a complex sparse tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static CTensorOld sub(CooCTensor src1, TensorOld src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CTensorOld dest = src2.mult(-1).toComplex();

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.shape.entriesIndex(src1.indices[i]);
            dest.entries[idx] = dest.entries[idx].add(src1.entries[i]);
        }

        return dest;
    }


    /**
     * Adds a scalar to a real sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static CTensorOld add(CooTensor src1, CNumber b) {
        CTensorOld sum = new CTensorOld(src1.shape, b);

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.shape.entriesIndex(src1.indices[i]);
            sum.entries[idx].add(src1.entries[i]);
        }

        return sum;
    }


    /**
     * Subtracts a scalar from a real sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static CTensorOld sub(CooTensor src1, CNumber b) {
        CTensorOld sum = new CTensorOld(src1.shape, b);

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.shape.entriesIndex(src1.indices[i]);
            sum.entries[idx].add(src1.entries[i]);
        }

        return sum;
    }


    /**
     * Adds a scalar to a real sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static CTensorOld sub(CooCTensor src1, double b) {
        CTensorOld sum = new CTensorOld(src1.shape, b);

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.shape.entriesIndex(src1.indices[i]);
            sum.entries[idx] = sum.entries[idx].sub(src1.entries[i]);
        }

        return sum;
    }


    /**
     * Adds a scalar to a real sparse COO tensor.
     * @param src1 Sparse tensor in sum.
     * @param b Scalar in sum.
     * @return A dense tensor which is the sum of {@code src1} and {@code b} such that {@code b} is added to each element of {@code
     * src1}.
     */
    public static CTensorOld add(CooCTensor src1, double b) {
        CTensorOld sum = new CTensorOld(src1.shape, b);

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.shape.entriesIndex(src1.indices[i]);
            sum.entries[idx] = sum.entries[idx].add(src1.entries[i]);
        }

        return sum;
    }
}