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

package org.flag4j.operations.dense_sparse.csr.real_complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

/**
 * This class contains low-level operations which act on a real/complex dense and a complex/real
 * sparse {@link CsrCMatrix CSR matrix}.
 */
public final class RealComplexCsrDenseOperations {

    private RealComplexCsrDenseOperations() {
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
    public static CMatrix applyBinOpp(CsrCMatrix src1, Matrix src2,
                                      BiFunction<Complex128, Double, Complex128> opp,
                                      UnaryOperator<Double> uOpp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);  // Ensure both matrices are same shape.

        Complex128[] dest;
        if(uOpp == null) dest = ArrayUtils.wrapAsComplex128(src2.entries, null);
        else dest = ArrayUtils.applyTransform(src2.entries.clone(), (Double a)->new Complex128(uOpp.apply(a)));

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];
                dest[idx] = opp.apply((Complex128) src1.entries[j], ((Complex128) dest[idx]).re);
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
    public static CMatrix applyBinOpp(Matrix src1, CsrCMatrix src2, BiFunction<Double, Complex128, Complex128> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        Complex128[] dest = ArrayUtils.wrapAsComplex128(src1.entries, null);

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[i];

                dest[idx] = opp.apply(src1.entries[idx], (Complex128) src2.entries[j]);
            }
        }

        return new CMatrix(src2.shape, dest);
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
    public static CMatrix applyBinOpp(CsrMatrix src1, CMatrix src2,
                                      BiFunction<Double, Complex128, Complex128> opp,
                                      UnaryOperator<Complex128> uOpp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.
        Field<Complex128>[] dest;

        if(uOpp == null) dest = src2.entries.clone();
        else dest = ArrayUtils.applyTransform(src2.entries.clone(), uOpp);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];
                dest[idx] = opp.apply(src1.entries[j], (Complex128) dest[idx]);
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
    public static CMatrix applyBinOpp(CMatrix src1, CsrMatrix src2, BiFunction<Complex128, Double, Complex128> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.
        Field<Complex128>[] dest = new Complex128[src2.entries.length];

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[i];
                dest[idx] = opp.apply((Complex128) src1.entries[idx], src2.entries[j]);
            }
        }

        return new CMatrix(src2.shape, dest);
    }


    /**
     * Applies an element-wise binary operation to a real dense and real sparse CSR matrix under the
     * assumption that {@code opp.apply(x, 0d) = 0d} and {@code opp.apply(0d, x) = 0d}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static CsrCMatrix applyBinOppToSparse(Matrix src1, CsrCMatrix src2, BiFunction<Double, Complex128, Complex128> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        int[] rowPointers = src2.rowPointers.clone();
        int[] colIndices = src2.colIndices.clone();
        Complex128[] entries = new Complex128[src2.entries.length];

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[j];
                entries[idx] = opp.apply(src1.entries[idx], (Complex128) src2.entries[j]);
            }
        }

        return new CsrCMatrix(src1.shape, entries, rowPointers, colIndices);
    }


    /**
     * Applies an element-wise binary operation to a real dense and real sparse CSR matrix under the
     * assumption that {@code opp.apply(x, 0d) = 0d} and {@code opp.apply(0d, x) = 0d}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static CsrCMatrix applyBinOppToSparse(CMatrix src1, CsrMatrix src2, BiFunction<Complex128, Double, Complex128> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        int[] rowPointers = src2.rowPointers.clone();
        int[] colIndices = src2.colIndices.clone();
        Complex128[] entries = new Complex128[src2.entries.length];

        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[j];
                entries[idx] = opp.apply((Complex128) src1.entries[idx], src2.entries[j]);
            }
        }

        return new CsrCMatrix(src1.shape, entries, rowPointers, colIndices);
    }


    /**
     * Applies an element-wise binary operation to a real sparse and complex dense CSR matrix under the
     * assumption that {@code opp.apply(0d, x) = 0d} where {@code x} is a {@link Complex128}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static CsrCMatrix applyBinOppToSparse(CsrMatrix src1, CMatrix src2, BiFunction<Double, Complex128, Complex128> opp) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        int[] rowPointers = src1.rowPointers.clone();
        int[] colIndices = src1.colIndices.clone();
        Complex128[] entries = new Complex128[src1.entries.length];

        for(int i=0; i<src1.numRows; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int src2RowOffset = i*src2.numCols;

            for(int j=start; j<stop; j++)
                entries[j] = opp.apply(src1.entries[j], (Complex128) src2.entries[src2RowOffset + src1.colIndices[j]]);
        }

        return new CsrCMatrix(src1.shape, entries, rowPointers, colIndices);
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
    public static CMatrix applyBinOpp(CsrMatrix src1, Complex128 b,
                                         BiFunction<Double, Complex128, Complex128> opp,
                                         UnaryOperator<Complex128> uOpp) {
        Complex128[] dest = new Complex128[src1.totalEntries().intValueExact()];
        if(uOpp != null) b = uOpp.apply(b);  // Apply unary operator if specified.
        Arrays.fill(dest, b);

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[j];

                dest[idx] = opp.apply(src1.entries[j], dest[idx]);
            }
        }

        return new CMatrix(src1.shape, dest);
    }


    /**
     * Computes the element-wise sum of two matrices.
     * @param a First matrix in sum.
     * @param b Second matrix in sum.
     * @return The element-wise sum of {@code a} and {@code b}.
     */
    public static CMatrix add(CsrCMatrix a, Matrix b) {
        return applyBinOpp(a, b, Complex128::add, null);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CMatrix sub(CsrCMatrix a, Matrix b) {
        return applyBinOpp(a, b, Complex128::add, (Double x)->-x);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CMatrix sub(Matrix a, CsrCMatrix b) {
        return applyBinOpp(a, b, (Double x, Complex128 y)->y.sub(x));
    }


    /**
     * Computes the element-wise sum of two matrices.
     * @param a First matrix in sum.
     * @param b Second matrix in sum.
     * @return The element-wise sum of {@code a} and {@code b}.
     */
    public static CMatrix add(CsrMatrix a, CMatrix b) {
        return applyBinOpp(a, b, (Double x, Complex128 y)->new Complex128(x+y.re, y.im), null);
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CMatrix sub(CsrMatrix a, CMatrix b) {
        return applyBinOpp(a, b, (Double x, Complex128 y)->y.add(x), (Complex128 x)->x.addInv());
    }


    /**
     * Computes the element-wise difference of two matrices.
     * @param a First matrix in difference.
     * @param b Second matrix in difference.
     * @return The element-wise difference of {@code a} and {@code b}.
     */
    public static CMatrix sub(CMatrix a, CsrMatrix b) {
        return applyBinOpp(a, b, (Complex128 x, Double y)->new Complex128(x.re-y, -x.im));
    }
}
