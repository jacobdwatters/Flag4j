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

package org.flag4j.operations.dense_sparse.csr.complex;


import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/**
 * This class contains low-level operations which act on a complex dense and a complex sparse {@link CsrCMatrix CSR matrix}.
 */
public final class ComplexCsrDenseOperations {

    private ComplexCsrDenseOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @param uOpp Unary operator for use with binary operations which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrix applyBinOpp(CsrCMatrix src1, CMatrix src2,
                                      BinaryOperator<Complex128> opp,
                                      UnaryOperator<Complex128> uOpp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        Field<Complex128>[] dest;
        if(uOpp == null) dest = Arrays.copyOf(src2.entries, src2.entries.length);
        else dest = ArrayUtils.applyTransform(src2.entries, uOpp);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];
                dest[idx] = opp.apply((Complex128) src1.entries[j], (Complex128) dest[idx]);
            }
        }

        return new CMatrix(src2.shape, dest);
    }



    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrix applyBinOpp(CMatrix src1, CsrCMatrix src2, BinaryOperator<Complex128> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        // TODO: Subtracting a sparse matrix from a dense matrix does not require a unary operator.
        //  Ensure that no method of this form requires this.

        Field<Complex128>[] dest = Arrays.copyOf(src2.entries, src2.entries.length);

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[i];
                dest[idx] = opp.apply((Complex128) src1.entries[idx], (Complex128) src2.entries[j]);
            }
        }

        return new CMatrix(src2.shape, dest);
    }


    /**
     * Applies the specified binary operator element-wise to a matrix and a scalar.
     * @param src1 First matrix in element-wise binary operation.
     * @param b Scalar to apply element-wise using the specified operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @param uOpp Unary operator for use with binary operations which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrix applyBinOpp(CsrCMatrix src1, double b,
                                         BinaryOperator<Complex128> opp,
                                         UnaryOperator<Double> uOpp) {
        Field<Complex128>[] dest = new Complex128[src1.totalEntries().intValueExact()];

        // Apply unary operator if specified.
        if(uOpp != null) ArrayUtils.fill(dest, uOpp.apply(b));
        else ArrayUtils.fill(dest, b);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];

                dest[idx] = opp.apply((Complex128) src1.entries[j], (Complex128) dest[idx]);
            }
        }

        return new CMatrix(src1.shape, dest);
    }


    /**
     * Applies the specified binary operator element-wise to a matrix and a scalar.
     * @param src1 First matrix in element-wise binary operation.
     * @param b Scalar to apply element-wise using the specified operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @param uOpp Unary operator for use with binary operations which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrix applyBinOpp(CsrCMatrix src1, Complex128 b,
                                         BinaryOperator<Complex128> opp,
                                         UnaryOperator<Complex128> uOpp) {
        Field<Complex128>[] dest = new Complex128[src1.totalEntries().intValueExact()];

        // Apply unary operator if specified.
        if(uOpp != null) Arrays.fill(dest, uOpp.apply(b));
        else Arrays.fill(dest, b);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];
                dest[idx] = opp.apply((Complex128) src1.entries[j], (Complex128) dest[idx]);
            }
        }

        return new CMatrix(src1.shape, dest);
    }


    /**
     * Applies an element-wise binary operation to a real dense and real sparse CSR matrix under the
     * assumption that {@code opp.apply(x, 0d) = 0d} and {@code opp.apply(0d, x) = 0d}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static CsrCMatrix applyBinOppToSparse(CMatrix src1, CsrCMatrix src2, BinaryOperator<Complex128> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        int[] rowPointers = src2.rowPointers.clone();
        int[] colIndices = src2.colIndices.clone();
        Field<Complex128>[] entries = new Complex128[src2.entries.length];

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[j];
                entries[idx] = opp.apply((Complex128) src1.entries[idx], (Complex128) src2.entries[j]);
            }
        }

        return new CsrCMatrix(src1.shape, entries, rowPointers, colIndices);
    }


    /**
     * Computes the element-wise sum of two matrices.
     * @param a First matrix in sum.
     * @param b Second matrix in sum.
     * @return The element-wise sum of {@code a} and {@code b}.
     */
    public static CMatrix add(CsrCMatrix a, CMatrix b) {
        return applyBinOpp(a, b, (Complex128 x, Complex128 y)->x.add(y), null);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CMatrix sub(CsrCMatrix a, CMatrix b) {
        return applyBinOpp(a, b, (Complex128 x, Complex128 y)->x.add(y), (Complex128 x)->x.addInv());
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CMatrix sub(CMatrix a, CsrCMatrix b) {
        return applyBinOpp(a, b, (Complex128 x, Complex128 y)->y.sub(x));
    }
}
