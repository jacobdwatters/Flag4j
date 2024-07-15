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

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.operations.common.complex.ComplexOperations;
import org.flag4j.operations.common.real.RealOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;


/**
 * This class provides low level methods for computing operations between a real/complex dense/sparse vector and a
 * complex/real sparse/dense vector.
 */
public class RealComplexDenseSparseVectorOperations {


    private RealComplexDenseSparseVectorOperations() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
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
    public static CNumber inner(double[] src1, CNumber[] src2, int[] indices, int sparseSize) {
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
     * Computes the vector inner product between a complex dense vector and a real sparse vector.
     * @param src1 Entries of the dense vector.
     * @param src2 Non-zero entries of the sparse vector.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param sparseSize The size of the sparse vector (including zero entries).
     * @return The inner product of the two vectors.
     * @throws IllegalArgumentException If the number of entries in the two vectors is not equivalent.
     */
    public static CNumber inner(CNumber[] src1, double[] src2, int[] indices, int sparseSize) {
        ParameterChecks.assertArrayLengthsEq(src1.length, sparseSize);
        CNumber innerProd = new CNumber();
        int index;

        for(int i=0; i<src2.length; i++) {
            index = indices[i];
            innerProd.addEq(src1[index].mult(src2[i]));
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
    public static CNumber inner(double[] src1, int[] indices, int sparseSize, CNumber[] src2) {
        ParameterChecks.assertArrayLengthsEq(src2.length, sparseSize);
        CNumber innerProd = new CNumber();
        int index;

        for(int i=0; i<src1.length; i++) {
            index = indices[i];
            innerProd.addEq(src2[index].conj().mult(src1[i]));
        }

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
    public static CNumber[] outerProduct(double[] src1, CNumber[] src2, int[] indices, int sparseSize) {
        CNumber[] dest = new CNumber[src1.length*sparseSize];
        ArrayUtils.fillZeros(dest);
        int index;

        for(int i=0; i<src1.length; i++) {
            for(int j=0; j<src2.length; j++) {
                index = indices[j];

                dest[i*src1.length + index] = src2[j].conj().mult(src1[i]);
            }
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
    public static CNumber[] outerProduct(CNumber[] src1, double[] src2, int[] indices, int sparseSize) {
        CNumber[] dest = new CNumber[sparseSize*src1.length];
        ArrayUtils.fillZeros(dest);
        int index;

        for(int i=0; i<src1.length; i++) {
            for(int j=0; j<src2.length; j++) {
                index = indices[j];
                dest[i*sparseSize + index] = src1[i].mult(src2[j]);
            }
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
    public static CNumber[] outerProduct(double[] src1, int[] indices, int sparseSize, CNumber[] src2) {
        ParameterChecks.assertEquals(sparseSize, src2.length);

        CNumber[] dest = new CNumber[src2.length*sparseSize];
        ArrayUtils.fillZeros(dest);
        int destIndex;

        for(int i=0; i<src1.length; i++) {
            destIndex = indices[i]*src2.length;

            for(CNumber v : src2) {
                dest[destIndex++] = v.mult(src1[i]);
            }
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
    public static CNumber[] outerProduct(CNumber[] src1, int[] indices, int sparseSize, double[] src2) {
        CNumber[] dest = new CNumber[sparseSize*src2.length];
        ArrayUtils.fillZeros(dest);
        int index;

        for(int i=0; i<src2.length; i++) {
            for(int j=0; j<src1.length; j++) {
                index = indices[j];
                dest[i*sparseSize + index] = src1[j].mult(src2[i]);
            }
        }

        return dest;
    }


    /**
     * Adds a real dense matrix to a complex sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the vector addition.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static CVector add(Vector src1, CooCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CVector dest = new CVector(src1);
        int index;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            index = src2.indices[i];
            dest.entries[index].addEq(src2.entries[i]);
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
    public static CVector add(CVector src1, CooVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CVector dest = new CVector(src1);
        int index;

        for(int i=0; i<src2.entries.length; i++) {
            index = src2.indices[i];
            dest.entries[index].addEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a complex sparse vector from a real dense vector.
     * @param src1 First vector.
     * @param src2 Second vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static CVector sub(Vector src1, CooCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CVector dest = new CVector(src1);
        int index;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
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
    public static CVector sub(CooCVector src1, Vector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CVector dest = new CVector(RealOperations.scalMult(src2.entries, -1));

        for(int i=0; i<src1.nonZeroEntries(); i++) {
            dest.entries[src1.indices[i]].addEq(src1.entries[i]);
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
    public static CVector sub(CooVector src1, CVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CVector dest = new CVector(ComplexOperations.scalMult(src2.entries, -1));

        for(int i=0; i<src1.nonZeroEntries(); i++) {
            dest.entries[src1.indices[i]].addEq(src1.entries[i]);
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
    public static CooCVector elemMult(Vector src1, CooCVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] entries = new CNumber[src2.entries.length];

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            entries[i] = src2.entries[i].mult(src1.entries[src2.indices[i]]);
        }

        return new CooCVector(src1.size, entries, src2.indices.clone());
    }


    /**
     * Subtracts a real sparse vector from a complex dense vector.
     * @param src1 First vector.
     * @param src2 Second vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static CVector sub(CVector src1, CooVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CVector dest = new CVector(src1);
        int index;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            index = src2.indices[i];
            dest.entries[index].subEq(src2.entries[i]);
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
    public static void addEq(CVector src1, CooVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int index;
        for(int i=0; i<src2.nonZeroEntries(); i++) {
            index = src2.indices[i];
            src1.entries[index].addEq(src2.entries[i]);
        }
    }


    /**
     * Computes the vector subtraction between a dense complex vector and a real sparse vector. The result is stored in
     * the first vector.
     * @param src1 First vector in subtraction.
     * @param src2 Second vector in subtraction.
     */
    public static void subEq(CVector src1, CooVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int index;
        for(int i=0; i<src2.nonZeroEntries(); i++) {
            index = src2.indices[i];
            src1.entries[index].subEq(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise multiplication of a complex dense vector with a real sparse vector.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the element-wise multiplication.
     * @throws IllegalArgumentException If the two vectors are not the same size.
     */
    public static CooCVector elemMult(CVector src1, CooVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] entries = new CNumber[src2.entries.length];

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            entries[i] = src1.entries[src2.indices[i]].mult(src2.entries[i]);
        }

        return new CooCVector(src1.size, entries, src2.indices.clone());
    }


    /**
     * Compute the element-wise division between a sparse vector and a dense vector.
     * @param src1 First vector in the element-wise division.
     * @param src2 Second vector in the element-wise division.
     * @return The result of the element-wise vector division.
     */
    public static CooCVector elemDiv(CooCVector src1, Vector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CNumber[] dest = new CNumber[src1.entries.length];

        for(int i=0; i<src1.entries.length; i++) {
            dest[i] = src1.entries[i].div(src2.entries[src1.indices[i]]);
        }

        return new CooCVector(src1.size, dest, src1.indices.clone());
    }


    /**
     * Compute the element-wise division between a sparse vector and a dense vector.
     * @param src1 First vector in the element-wise division.
     * @param src2 Second vector in the element-wise division.
     * @return The result of the element-wise vector division.
     */
    public static CooCVector elemDiv(CooVector src1, CVector src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);
        CNumber[] dest = new CNumber[src1.entries.length];

        for(int i=0; i<src1.entries.length; i++) {
            dest[i] = new CNumber(src1.entries[i]).div(src2.entries[src1.indices[i]]);
        }

        return new CooCVector(src1.size, dest, src1.indices.clone());
    }
}
