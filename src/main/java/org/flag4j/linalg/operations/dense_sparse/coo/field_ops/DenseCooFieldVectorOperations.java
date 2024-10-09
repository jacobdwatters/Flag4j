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

package org.flag4j.linalg.operations.dense_sparse.coo.field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.CooFieldVectorBase;
import org.flag4j.arrays.backend.DenseFieldVectorBase;
import org.flag4j.linalg.operations.common.field_ops.FieldOperations;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * This class provides low level methods for computing operations between dense/sparse and
 * sparse/dense field vectors.
 */
public final class DenseCooFieldVectorOperations {

    private DenseCooFieldVectorOperations() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the vector inner product between a complex dense vector and a complex sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of nonzero values in sparse vector.
     * @param sparseSize Full size of the sparse vector (i.e. total number of entries including zeros).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of entries in the two vectors is not equivalent.
     */
    public static <T extends Field<T>> Field<T> innerProduct(Field<T>[] src1, Field<T>[] src2,
                                                             int[] indices, int sparseSize) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, sparseSize);
        T innerProd = (src1.length > 0) ? src1[0].getZero() : null;

        for(int i=0; i<src2.length; i++) {
            int index = indices[i];
            innerProd = innerProd.add(src2[i].conj().mult((T) src1[index]));
        }

        return innerProd;
    }


    /**
     * Computes the vector inner product between a complex dense vector and a complex sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of nonzero values in sparse vector.
     * @param sparseSize Full size of the sparse vector (i.e. total number of entries including zeros).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of entries in the two vectors is not equivalent.
     */
    public static <T extends Field<T>> Field<T> innerProduct(Field<T>[] src1, int[] indices, int sparseSize, Field<T>[] src2) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, sparseSize);
        T innerProd = (src1.length > 0) ? src1[0].getZero() : null;

        for(int i=0; i<src1.length; i++) {
            int index = indices[i];
            innerProd = innerProd.add(src1[i].mult(src2[index].conj()));
        }

        return innerProd;
    }


    /**
     * Computes the vector outer product between a complex dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries of sparse vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static <T extends Field<T>> Field<T>[] outerProduct(Field<T>[] src1, Field<T>[] src2, int[] indices, int sparseSize) {
        Field<T>[] dest = new Field[src1.length*sparseSize];
        Arrays.fill(dest, (src1.length > 0) ? src1[0].getZero() : null);
        int index;

        for(int i=0; i<src1.length; i++) {
            for(int j=0; j<src2.length; j++) {
                index = indices[j];
                dest[i*sparseSize + index] = src1[i].mult(src2[j].conj());
            }
        }

        return dest;
    }


    /**
     * Computes the vector outer product between a complex dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries of sparse vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static <T extends Field<T>> Field<T>[] outerProduct(Field<T>[] src2, int[] indices, int sparseSize, Field<T>[] src1) {
        ValidateParameters.ensureEquals(sparseSize, src2.length);
        Field<T>[] dest = new Field[src2.length*sparseSize];

        for(int i=0; i<src1.length; i++) {
            int destIndex = indices[i]*src2.length;

            for(Field<T> v : src2)
                dest[destIndex++] = src1[i].mult((T) v);
        }

        return dest;
    }


    /**
     * Computes the element-wise addition between a dense complex vector and sparse complex vectors.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the vector addition.
     */
    public static <T extends Field<T>> DenseFieldVectorBase<?, ?, ?, T> add(DenseFieldVectorBase<?, ?, ?, T> src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldVectorBase<?, ?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.indices[i];
            dest.entries[idx] = dest.entries[idx].add((T) src2.entries[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise addition between a dense complex vector and sparse complex vectors.
     * The result is stored in the first vector.
     * @param src1 Dense vector. Modified.
     * @param src2 Sparse vector.
     */
    public static <T extends Field<T>> void addEq(DenseFieldVectorBase<?, ?, ?, T> src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.indices[i];
            src1.entries[idx] = src1.entries[idx].add((T) src2.entries[i]);
        }
    }



    /**
     * Subtracts a complex sparse vector from a complex dense vector.
     * @param src1 First vector.
     * @param src2 Second vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static <T extends Field<T>> DenseFieldVectorBase<?, ?, ?, T> sub(
            DenseFieldVectorBase<?, ?, ?, T> src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldVectorBase<?, ?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.entries.length; i++) {
            int index = src2.indices[i];
            dest.entries[index] = dest.entries[index].sub((T) src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a complex dense vector from a complex sparse vector.
     * @param src1 Sparse vector.
     * @param src2 Dense vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static <T extends Field<T>> DenseFieldVectorBase<?, ?, ?, T> sub(
            CooFieldVectorBase<?, ?, ?, ?, T> src1, DenseFieldVectorBase<?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldVectorBase<?, ?, ?, T> dest = src2.makeLikeTensor(FieldOperations.scalMult(src2.entries, -1));

        for(int i=0, size=src1.nnz; i<size; i++) {
            int idx = src1.indices[i];
            dest.entries[idx] = dest.entries[idx].add((T) src1.entries[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise subtraction between a dense complex vector and sparse complex vectors.
     * The result is stored in the first vector.
     * @param src1 Dense vector. Modified.
     * @param src2 Sparse vector.
     */
    public static <T extends Field<T>> void subEq(DenseFieldVectorBase<?, ?, ?, T> src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.indices[i];
            src1.entries[idx] = src1.entries[idx].sub((T) src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise multiplication of a complex dense vector with a complex sparse vector.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the element-wise multiplication.
     * @throws IllegalArgumentException If the two vectors are not the same size.
     */
    public static <T extends Field<T>> CooFieldVectorBase<?, ?, ?, ?, T> elemMult(
            DenseFieldVectorBase<?, ?, ?, T> src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Field<T>[] entries = new Field[src2.entries.length];

        for(int i=0; i<src2.nnz; i++)
            entries[i] = src1.entries[src2.indices[i]].mult((T) src2.entries[i]);

        return src2.makeLikeTensor(src1.size, entries, src2.indices.clone());
    }


    /**
     * Compute the element-wise division between a complex sparse vector and a complex dense vector.
     * @param src1 First vector in the element-wise division.
     * @param src2 Second vector in the element-wise division.
     * @return The result of the element-wise vector division.
     */
    public static <T extends Field<T>> CooFieldVectorBase<?, ?, ?, ?, T> elemDiv(
            CooFieldVectorBase<?, ?, ?, ?, T> src1, DenseFieldVectorBase<?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Field<T>[] dest = new Field[src1.entries.length];

        for(int i=0; i<src1.nnz; i++)
            dest[i] = src1.entries[i].div((T) src2.entries[src1.indices[i]]);

        return src1.makeLikeTensor(src1.size, dest, src1.indices.clone());
    }
}
