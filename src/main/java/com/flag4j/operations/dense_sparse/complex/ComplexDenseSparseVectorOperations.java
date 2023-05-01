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

package com.flag4j.operations.dense_sparse.complex;


import com.flag4j.CVector;
import com.flag4j.SparseCVector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class provides low level methods for computing operations between complex dense/sparse and complex
 * sparse/dense vectors.
 */
public class ComplexDenseSparseVectorOperations {

    private ComplexDenseSparseVectorOperations() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
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
    public static CNumber innerProduct(CNumber[] src1, CNumber[] src2, int[] indices, int sparseSize) {
        ParameterChecks.assertArrayLengthsEq(src1.length, sparseSize);
        CNumber innerProd = new CNumber();
        int index;

        for(int i=0; i<src2.length; i++) {
            index = indices[i];
            innerProd.addEq(src2[i].conj().mult(src1[index]));
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
    public static CNumber[] outerProduct(CNumber[] src1, CNumber[] src2, int[] indices, int sparseSize) {
        CNumber[] dest = new CNumber[src1.length*sparseSize];
        ArrayUtils.fillZeros(dest);
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
     * Computes the element-wise addition between a dense complex vector and sparse complex vectors.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the vector addition.
     */
    public static CVector add(CVector src1, SparseCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CVector dest = new CVector(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            dest.entries[src2.indices[i]].addEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise addition between a dense complex vector and sparse complex vectors.
     * The result is stored in the first vector.
     * @param src1 Dense vector. Modified.
     * @param src2 Sparse vector.
     */
    public static void addEq(CVector src1, SparseCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src2.indices[i]].addEq(src2.entries[i]);
        }
    }



    /**
     * Subtracts a complex sparse vector from a complex dense vector.
     * @param src1 First vector.
     * @param src2 Second vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static CVector sub(CVector src1, SparseCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CVector dest = src1.copy();
        int index;

        for(int i=0; i<src2.entries.length; i++) {
            index = src2.indices[i];
            dest.entries[index].subEq(src2.entries[i]);
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
    public static CVector sub(SparseCVector src1, CVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CVector dest = new CVector(ComplexOperations.scalMult(src2.entries, -1));

        for(int i=0; i<src1.nonZeroEntries(); i++) {
            dest.entries[src1.indices[i]].addEq(src1.entries[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise subtraction between a dense complex vector and sparse complex vectors.
     * The result is stored in the first vector.
     * @param src1 Dense vector. Modified.
     * @param src2 Sparse vector.
     */
    public static void subEq(CVector src1, SparseCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            src1.entries[src2.indices[i]].subEq(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise multiplication of a complex dense vector with a complex sparse vector.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the element-wise multiplication.
     * @throws IllegalArgumentException If the two vectors are not the same size.
     */
    public static SparseCVector elemMult(CVector src1, SparseCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] entries = new CNumber[src2.entries.length];

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            entries[i] = src1.entries[src2.indices[i]].mult(src2.entries[i]);
        }

        return new SparseCVector(src1.size, entries, src2.indices.clone());
    }
}
