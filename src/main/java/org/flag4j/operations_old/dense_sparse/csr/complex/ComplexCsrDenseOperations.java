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

package org.flag4j.operations_old.dense_sparse.csr.complex;


import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.sparse.CsrCMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/**
 * This class contains low-level operations_old which act on a complex dense and a complex sparse {@link CsrCMatrixOld CSR matrix}.
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
     * @param uOpp Unary operator for use with binary operations_old which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrixOld applyBinOpp(CsrCMatrixOld src1, CMatrixOld src2,
                                         BinaryOperator<CNumber> opp,
                                         UnaryOperator<CNumber> uOpp) {
        ParameterChecks.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        CNumber[] dest;
        if(uOpp == null) dest = Arrays.copyOf(src2.entries, src2.entries.length);
        else dest = ArrayUtils.applyTransform(src2.entries, uOpp);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];

                dest[idx] = opp.apply(
                        src1.entries[j],
                        dest[idx]
                );
            }
        }

        return new CMatrixOld(src2.shape, dest);
    }



    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrixOld applyBinOpp(CMatrixOld src1, CsrCMatrixOld src2, BinaryOperator<CNumber> opp) {
        ParameterChecks.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        // TODO: Subtracting a sparse matrix from a dense matrix does not require a unary operator.
        //  Ensure that no method of this form requires this.

        CNumber[] dest = Arrays.copyOf(src2.entries, src2.entries.length);

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[i];

                dest[idx] = opp.apply(
                        src1.entries[idx],
                        src2.entries[j]
                );
            }
        }

        return new CMatrixOld(src2.shape, dest);
    }


    /**
     * Applies the specified binary operator element-wise to a matrix and a scalar.
     * @param src1 First matrix in element-wise binary operation.
     * @param b Scalar to apply elementwise using the specified operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @param uOpp Unary operator for use with binary operations_old which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrixOld applyBinOpp(CsrCMatrixOld src1, double b,
                                         BinaryOperator<CNumber> opp,
                                         UnaryOperator<Double> uOpp) {
        CNumber[] dest = new CNumber[src1.totalEntries().intValueExact()];

        // Apply unary operator if specified.
        if(uOpp != null) ArrayUtils.fill(dest, uOpp.apply(b));
        else ArrayUtils.fill(dest, b);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];

                dest[idx] = opp.apply(
                        src1.entries[j],
                        dest[idx]
                );
            }
        }

        return new CMatrixOld(src1.shape, dest);
    }


    /**
     * Applies the specified binary operator element-wise to a matrix and a scalar.
     * @param src1 First matrix in element-wise binary operation.
     * @param b Scalar to apply elementwise using the specified operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @param uOpp Unary operator for use with binary operations_old which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrixOld applyBinOpp(CsrCMatrixOld src1, CNumber b,
                                         BinaryOperator<CNumber> opp,
                                         UnaryOperator<CNumber> uOpp) {
        CNumber[] dest = new CNumber[src1.totalEntries().intValueExact()];

        // Apply unary operator if specified.
        if(uOpp != null) Arrays.fill(dest, uOpp.apply(b));
        else Arrays.fill(dest, b);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];

                dest[idx] = opp.apply(
                        src1.entries[j],
                        dest[idx]
                );
            }
        }

        return new CMatrixOld(src1.shape, dest);
    }


    /**
     * Applies an element-wise binary operation to a real dense and real sparse CSR matrix under the
     * assumption that {@code opp.apply(x, 0d) = 0d} and {@code opp.apply(0d, x) = 0d}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static CsrCMatrixOld applyBinOppToSparse(CMatrixOld src1, CsrCMatrixOld src2, BinaryOperator<CNumber> opp) {
        ParameterChecks.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        int[] rowPointers = src2.rowPointers.clone();
        int[] colIndices = src2.colIndices.clone();
        CNumber[] entries = new CNumber[src2.entries.length];

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[j];

                entries[idx] = opp.apply(src1.entries[idx], src2.entries[j]);
            }
        }

        return new CsrCMatrixOld(src1.shape, entries, rowPointers, colIndices);
    }
}
