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

package org.flag4j.linalg.operations.dense_sparse.coo.real_complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.operations.common.real.RealOperations;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;


/**
 * This class provides low level methods for computing operations between a real/complex dense/sparse vector and a
 * complex/real sparse/dense vector.
 */
public final class RealComplexDenseSparseVectorOperations {


    private RealComplexDenseSparseVectorOperations() {
        // Hide default constructor in utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Adds a real dense matrix to a complex sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the vector addition.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static CVector add(Vector src1, CooCVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        CVector dest = src1.toComplex();
        int index;

        for(int i=0; i<src2.nnz; i++) {
            index = src2.indices[i];
            dest.entries[index] = dest.entries[index].add((Complex128) src2.entries[i]);
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
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        CVector dest = src1.toComplex();
        int index;

        for(int i=0; i<src2.nnz; i++) {
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
    public static CVector sub(CooCVector src1, Vector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        CVector dest = new Vector(RealOperations.scalMult(src2.entries, -1)).toComplex();

        for(int i=0; i<src1.nnz; i++) {
            int idx = src1.indices[i];
            dest.entries[idx] = dest.entries[idx].add((Complex128) src1.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse vector from a complex dense vector.
     * @param src1 First vector.
     * @param src2 Second vector.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the vectors do not have the same shape.
     */
    public static CVector sub(CVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        CVector dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int index = src2.indices[i];
            dest.entries[index] = dest.entries[index].sub(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Computes the element-wise multiplication of a complex dense vector with a real sparse vector.
     * @param src1 Dense vector.
     * @param src2 Sparse vector.
     * @return The result of the element-wise multiplication.
     * @throws IllegalArgumentException If the two vectors are not the same size.
     */
    public static CooCVector elemMult(CVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128[] entries = new Complex128[src2.entries.length];

        for(int i=0; i<src2.nnz; i++)
            entries[i] = src1.entries[src2.indices[i]].mult(src2.entries[i]);

        return new CooCVector(src1.size, entries, src2.indices.clone());
    }


    /**
     * Compute the element-wise division between a sparse vector and a dense vector.
     * @param src1 First vector in the element-wise division.
     * @param src2 Second vector in the element-wise division.
     * @return The result of the element-wise vector division.
     */
    public static CooCVector elemDiv(CooVector src1, CVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Complex128[] dest = new Complex128[src1.entries.length];

        for(int i=0; i<src1.entries.length; i++) {
            dest[i] = new Complex128(src1.entries[i]).div((Complex128) src2.entries[src1.indices[i]]);
        }

        return new CooCVector(src1.size, dest, src1.indices.clone());
    }
}
