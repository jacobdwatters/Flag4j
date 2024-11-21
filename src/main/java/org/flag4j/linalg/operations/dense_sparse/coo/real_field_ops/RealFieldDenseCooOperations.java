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

package org.flag4j.linalg.operations.dense_sparse.coo.real_field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.CooFieldTensorBase;
import org.flag4j.arrays.backend.DenseFieldTensorBase;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains methods to apply common binary operations to a real/field dense matrix and to a field/real sparse matrix.
 */
public final class RealFieldDenseCooOperations {

    private RealFieldDenseCooOperations() {
        // Hide default constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the element-wise multiplication between a real dense tensor and a complex sparse tensor.
     * @param src1 Real dense tensor.
     * @param src2 Complex sparse tensor.
     * @return The result of element-wise multiplication between the two tensors.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends Field<T>> CooFieldTensorBase<?, ?, T> elemMult(Tensor src1, CooFieldTensorBase<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        Field<T>[] destEntries = new Field[src2.nnz];
        int[][] destIndices = new int[src2.indices.length][src2.indices[0].length];
        ArrayUtils.deepCopy(src2.indices, destIndices);

        for(int i=0, size=destEntries.length; i<size; i++) {
            int index = src2.shape.getFlatIndex(src2.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src2.entries[i].mult(src1.entries[index]);
        }

        return src2.makeLikeTensor(src2.shape, destEntries, destIndices);
    }


    /**
     * Computes the element-wise division between a real dense tensor and a real sparse tensor.
     * @param src1 Real sparse tensor.
     * @param src2 Real dense tensor.
     * @return The result of element-wise division.
     * @throws IllegalArgumentException If the tensors do not have the same shape.
     */
    public static <T extends Field<T>> CooFieldTensorBase<?, ?, T> elemDiv(CooFieldTensorBase<?, ?, T> src1, Tensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Field<T>[] destEntries = new Field[src1.nnz];
        int[][] destIndices = new int[src1.indices.length][src1.indices[0].length];
        ArrayUtils.deepCopy(src1.indices, destIndices);

        for(int i=0, size=destEntries.length; i<size; i++) {
            int index = src2.shape.getFlatIndex(src1.indices[i]); // Get index of non-zero entry.
            destEntries[i] = src1.entries[index].div(src2.entries[i]);
        }

        return src1.makeLikeTensor(src2.shape, destEntries, destIndices);
    }


    /**
     * Adds a dense complex tensor to a real sparse tensor.
     * @param src1 Complex dense tensor.
     * @param src2 Real sparse tensor.
     * @return The result of the tensor addition.
     */
    public static <T extends Field<T>> DenseFieldTensorBase<?, ?, T> add(DenseFieldTensorBase<?, ?, T> src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldTensorBase<?, ?, T> dest = src1.copy();

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = dest.shape.getFlatIndex(src2.indices[i]);
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
    public static <T extends Field<T>> DenseFieldTensorBase<?, ?, T> sub(DenseFieldTensorBase<?, ?, T> src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldTensorBase<?, ?, T> dest = src1.copy();

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = dest.shape.getFlatIndex(src2.indices[i]);
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
    public static <T extends Field<T>> void addEq(DenseFieldTensorBase<?, ?, T> src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = src2.shape.getFlatIndex(src2.indices[i]);
            src1.entries[idx] = src1.entries[idx].add(src2.entries[i]);
        }
    }


    /**
     * Computes element-wise subtraction between a complex dense tensor and a real sparse tensor. The result is stored
     * in the complex dense tensor.
     * @param src1 The complex dense tensor. Also, the storage for the computation.
     * @param src2 The real sparse tensor.
     */
    public static <T extends Field<T>> void subEq(DenseFieldTensorBase<?, ?, T> src1, CooTensor src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0, size=src2.nnz; i<size; i++) {
            int idx = src2.shape.getFlatIndex(src2.indices[i]);
            src1.entries[idx] = src1.entries[idx].sub(src2.entries[i]);
        }
    }


    /**
     * Subtracts a complex dense tensor from a real sparse tensor.
     * @param src1 First tensor in the sum.
     * @param src2 Second tensor in the sum.
     * @return The result of the tensor addition.
     * @throws IllegalArgumentException If the tensors do not have the same shape.t
     */
    public static <T extends Field<T>> DenseFieldTensorBase<?, ?, T> sub(CooTensor src1, DenseFieldTensorBase<?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldTensorBase<?, ?, T> dest = src2.mult(-1);

        for(int i=0, size=src1.nnz; i<size; i++) {
            int idx = src1.shape.getFlatIndex(src1.indices[i]);
            dest.entries[idx] = dest.entries[idx].add(src1.entries[i]);
        }

        return dest;
    }
}
