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

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.operations.common.complex.Complex128Operations;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * This class provides low level methods for computing operations between complex dense/sparse and complex
 * sparse/dense vectors.
 */
public final class ComplexDenseSparseVectorOperations {

    private ComplexDenseSparseVectorOperations() {
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
    public static Complex128 innerProduct(Field<Complex128>[] src1, Field<Complex128>[] src2, int[] indices, int sparseSize) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, sparseSize);
        Complex128 innerProd = Complex128.ZERO;
        int index;

        for(int i=0; i<src2.length; i++) {
            index = indices[i];
            innerProd = innerProd.add(src2[i].conj().mult((Complex128) src1[index]));
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
    public static Complex128 innerProduct(Field<Complex128>[] src1, int[] indices, int sparseSize, Field<Complex128>[] src2) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, sparseSize);
        Complex128 innerProd = Complex128.ZERO;
        int index;

        for(int i=0; i<src1.length; i++) {
            index = indices[i];
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
    public static Complex128[] outerProduct(Field<Complex128>[] src1, Field<Complex128>[] src2, int[] indices, int sparseSize) {
        Complex128[] dest = new Complex128[src1.length*sparseSize];
        Arrays.fill(dest, Complex128.ZERO);
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
    public static Complex128[] outerProduct(Field<Complex128>[] src2, int[] indices, int sparseSize, Field<Complex128>[] src1) {
        ValidateParameters.ensureEquals(sparseSize, src2.length);

        Complex128[] dest = new Complex128[src2.length*sparseSize];
        int destIndex;

        for(int i=0; i<src1.length; i++) {
            destIndex = indices[i]*src2.length;

            for(Field<Complex128> v : src2) {
                dest[destIndex++] = src1[i].mult((Complex128) v);
            }
        }

        return dest;
    }


    /**
     * Computes the element-wise addition between a dense complex vector and sparse complex vectors.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the vector addition.
     */
    public static CVector add(CVector src1, CooCVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        CVector dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.indices[i];
            dest.entries[idx] = dest.entries[idx].add((Complex128) src2.entries[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise addition between a dense complex vector and sparse complex vectors.
     * The result is stored in the first vector.
     * @param src1 Dense vector. Modified.
     * @param src2 Sparse vector.
     */
    public static void addEq(CVector src1, CooCVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.indices[i];
            src1.entries[idx] = src1.entries[idx].add((Complex128) src2.entries[i]);
        }
    }



    /**
     * Subtracts a complex sparse vector from a complex dense vector.
     * @param src1 First vector.
     * @param src2 Second vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static CVector sub(CVector src1, CooCVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        CVector dest = src1.copy();
        int index;

        for(int i=0; i<src2.entries.length; i++) {
            index = src2.indices[i];
            dest.entries[index] = dest.entries[index].sub((Complex128) src2.entries[i]);
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
    public static CVector sub(CooCVector src1, CVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        CVector dest = new CVector(Complex128Operations.scalMult(src2.entries, -1));

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.indices[i];
            dest.entries[idx] = dest.entries[idx].add((Complex128) src1.entries[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise subtraction between a dense complex vector and sparse complex vectors.
     * The result is stored in the first vector.
     * @param src1 Dense vector. Modified.
     * @param src2 Sparse vector.
     */
    public static void subEq(CVector src1, CooCVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.indices[i];
            src1.entries[idx] = src1.entries[idx].sub((Complex128) src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise multiplication of a complex dense vector with a complex sparse vector.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the element-wise multiplication.
     * @throws IllegalArgumentException If the two vectors are not the same size.
     */
    public static CooCVector elemMult(CVector src1, CooCVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128[] entries = new Complex128[src2.entries.length];

        for(int i=0; i<src2.nnz; i++)
            entries[i] = src1.entries[src2.indices[i]].mult((Complex128) src2.entries[i]);

        return new CooCVector(src1.size, entries, src2.indices.clone());
    }


    /**
     * Compute the element-wise division between a complex sparse vector and a complex dense vector.
     * @param src1 First vector in the element-wise division.
     * @param src2 Second vector in the element-wise division.
     * @return The result of the element-wise vector division.
     */
    public static CooCVector elemDiv(CooCVector src1, CVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128[] dest = new Complex128[src1.entries.length];

        for(int i=0; i<src1.nnz; i++)
            dest[i] = src1.entries[i].div((Complex128) src2.entries[src1.indices[i]]);

        return new CooCVector(src1.size, dest, src1.indices.clone());
    }
}
