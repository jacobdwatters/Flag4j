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


import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.CooFieldVectorBase;
import org.flag4j.arrays.backend.DenseFieldVectorBase;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.operations.common.field_ops.FieldOperations;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * This class provides low level methods for computing operations between a real/field dense/sparse vector and a
 * field/real sparse/dense vector.
 */
public final class RealFieldDenseCooVectorOperations {

    private RealFieldDenseCooVectorOperations() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the vector inner product between a real dense vector and a complex sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param sparseSize The size of the sparse vector (including zero entries).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of entries in the two vectors is not equivalent.
     */
    public static <T extends Field<T>> Field<T> inner(double[] src1, Field<T>[] src2, int[] indices, int sparseSize) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, sparseSize);
        Field<T> innerProd = (src2.length > 0) ? src2[0].getZero() : null;

        for(int i=0; i<src2.length; i++) {
            int index = indices[i];
            innerProd = innerProd.add(src2[i].conj().mult(src1[index]));
        }

        return innerProd;
    }


    /**
     * Computes the vector inner product between a complex dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param sparseSize The size of the sparse vector (including zero entries).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of entries in the two vectors is not equivalent.
     */
    public static <T extends Field<T>> Field<T> inner(Field<T>[] src1, double[] src2, int[] indices, int sparseSize) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, sparseSize);
        Field<T> innerProd = (src1.length > 0) ? src1[0].getZero() : null;

        for(int i=0; i<src2.length; i++) {
            int index = indices[i];
            innerProd = innerProd.add(src1[index].mult(src2[i]));
        }

        return innerProd;
    }


    /**
     * Computes the vector inner product between a complex dense vector and a real sparse vector.
     * @param src1 Non-zero entries of the sparse vector.
     * @param src2 Entries of the dense vector.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param sparseSize The size of the sparse vector (including zero entries).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of entries in the two vectors is not equivalent.
     */
    public static <T extends Field<T>> Field<T> inner(double[] src1, int[] indices, int sparseSize, Field<T>[] src2) {
        ValidateParameters.ensureArrayLengthsEq(src2.length, sparseSize);
        Field<T> innerProd = (src2.length > 0) ? src2[0].getZero() : null;

        for(int i=0; i<src1.length; i++)
            innerProd = innerProd.add(src2[indices[i]].conj().mult(src1[i]));

        return innerProd;
    }


    /**
     * Computes the vector outer product between a real dense vector and a complex sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero entries.
     * @return The matrix resulting from the vector outer product.
     */
    public static <T extends Field<T>> Field<T>[] outerProduct(double[] src1, Field<T>[] src2, int[] indices, int sparseSize) {
        Field<T>[] dest = new Field[src1.length*sparseSize];
        Arrays.fill(dest, Complex128.ZERO);

        for(int i=0, size=src1.length; i<size; i++) {
            int destIdx = i*src1.length;
            double val1 = src1[i];

            for(int j=0, size2=src2.length; j<size2; j++)
                dest[destIdx + indices[j]] = src2[j].conj().mult(val1);
        }

        return dest;
    }


    /**
     * Computes the vector outer product between a complex dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero entries.
     * @return The matrix resulting from the vector outer product.
     */
    public static <T extends Field<T>> Field<T>[] outerProduct(Field<T>[] src1, double[] src2, int[] indices, int sparseSize) {
        Field<T>[] dest = new Field[sparseSize*src1.length];
        Arrays.fill(dest, Complex128.ZERO);

        for(int i=0, size=src1.length; i<size; i++) {
            int destIdx = i*sparseSize;
            Field<T> val1 = src1[i];

            for(int j=0, size2=src2.length; j<size2; j++)
                dest[destIdx + indices[j]] = val1.mult(src2[j]);
        }

        return dest;
    }


    /**
     * Computes the vector outer product between a complex dense vector and a real sparse vector.
     * @param src1 Non-zero entries of the real sparse vector.
     * @param indices Indices of non-zero entries of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero entries.
     * @param src2 Entries of the complex dense vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static <T extends Field<T>> Field<T>[] outerProduct(double[] src1, int[] indices, int sparseSize, Field<T>[] src2) {
        ValidateParameters.ensureEquals(sparseSize, src2.length);

        Field<T>[] dest = new Field[src2.length*sparseSize];
        Arrays.fill(dest, Complex128.ZERO);
        int destIndex;

        for(int i=0; i<src1.length; i++) {
            destIndex = indices[i]*src2.length;

            for(Field<T> v : src2)
                dest[destIndex++] = v.mult(src1[i]);
        }

        return dest;
    }


    /**
     * Computes the vector outer product between a rea; dense vector and a complex sparse vector.
     * @param src1 Non-zero entries of the complex sparse vector.
     * @param indices Indices of non-zero entries of sparse vector.
     * @param sparseSize Full size of the sparse vector including zero entries.
     * @param src2 Entries of the real dense vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static <T extends Field<T>> Field<T>[] outerProduct(Field<T>[] src1, int[] indices, int sparseSize, double[] src2) {
        Field<T>[] dest = new Field[sparseSize*src2.length];
        Arrays.fill(dest, Complex128.ZERO);

        for(int i=0; i<src2.length; i++) {
            int destIdx = i*sparseSize;

            for(int j=0; j<src1.length; j++)
                dest[destIdx + indices[j]] = src1[j].mult(src2[i]);
        }

        return dest;
    }


    /**
     * Adds a complex dense matrix to a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the vector addition.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static <T extends Field<T>> DenseFieldVectorBase<?, ?, ?, T> add(DenseFieldVectorBase<?, ?, ?, T> src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldVectorBase<?, ?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.entries.length; i++) {
            int index = src2.indices[i];
            dest.entries[index] = dest.entries[index].add(src2.entries[i]);
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
    public static <T extends Field<T>> DenseFieldVectorBase<?, ?, ?, T> sub(CooVector src1, DenseFieldVectorBase<?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldVectorBase<?, ?, ?, T> dest = src2.makeLikeTensor(FieldOperations.scalMult(src2.entries, -1));

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.indices[i];
            dest.entries[idx] = dest.entries[idx].add(src1.entries[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise multiplication of a real dense vector with a complex sparse vector.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the element-wise multiplication.
     * @throws IllegalArgumentException If the two vectors are not the same size.
     */
    public static <T extends Field<T>> CooFieldVectorBase<?, ?, ?, ?, T> elemMult(
            Vector src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Field<T>[] entries = new Field[src2.entries.length];

        for(int i=0, size=src2.nnz; i<size; i++)
            entries[i] = src2.entries[i].mult(src1.entries[src2.indices[i]]);

        return src2.makeLikeTensor(src1.size, entries, src2.indices.clone());
    }


    /**
     * Subtracts a real sparse vector from a complex dense vector.
     * @param src1 First vector.
     * @param src2 Second vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static <T extends Field<T>> DenseFieldVectorBase<?, ?, ?, T> sub(DenseFieldVectorBase<?, ?, ?, T> src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldVectorBase<?, ?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int index = src2.indices[i];
            dest.entries[index] = dest.entries[index].sub(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Computes the vector addition between a dense complex vector and a sparse real vector. The result is stored in
     * the first vector.
     * @param src1 First vector to add. Also, where the result is stored.
     * @param src2 Second vector to add.
     * @throws IllegalArgumentException If the vectors do not have the same size.
     */
    public static <T extends Field<T>> void addEq(DenseFieldVectorBase<?, ?, ?, T> src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int index = src2.indices[i];
            src1.entries[index] = src1.entries[index].add(src2.entries[i]);
        }
    }


    /**
     * Computes the vector subtraction between a dense complex vector and a real sparse vector. The result is stored in
     * the first vector.
     * @param src1 First vector in subtraction.
     * @param src2 Second vector in subtraction.
     */
    public static <T extends Field<T>> void subEq(DenseFieldVectorBase<?, ?, ?, T> src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int index = src2.indices[i];
            src1.entries[index] = src1.entries[index].sub(src2.entries[i]);
        }
    }


    /**
     * Compute the element-wise division between a sparse vector and a dense vector.
     * @param src1 First vector in the element-wise division.
     * @param src2 Second vector in the element-wise division.
     * @return The result of the element-wise vector division.
     */
    public static <T extends Field<T>> CooFieldVectorBase<?, ?, ?, ?, T> elemDiv(CooFieldVectorBase<?, ?, ?, ?, T> src1, Vector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Field<T>[] dest = new Field[src1.entries.length];

        for(int i=0, size=src1.entries.length; i<size; i++)
            dest[i] = src1.entries[i].div(src2.entries[src1.indices[i]]);

        return src1.makeLikeTensor(src1.size, dest, src1.indices.clone());
    }
}
