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

package org.flag4j.operations.sparse.csr.real_complex;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CsrCMatrixOld;
import org.flag4j.arrays_old.sparse.CsrMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

/**
 * This class contains low-level implementations for element-wise operations_old on real/complex CSR matrices.
 */
public final class RealComplexCsrOperations {

    private RealComplexCsrOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Applies an element-wise binary operation to two {@link CsrMatrixOld CSR Matrices}. <br><br>
     *
     * Note, this methods efficiency relies heavily on the assumption that both operand matrices are very large and very
     * sparse. If the two matrices are not large and very sparse, this method will likely be
     * significantly slower than simply converting the matrices to {@link MatrixOld dense matrices} and using a dense
     * matrix addition algorithm.
     * @param src1 The first matrix in the operation.
     * @param src2 The second matrix in the operation.
     * @param opp Binary operator to apply element-wise to <code>src1</code> and <code>src2</code>.
     * @param uOpp Unary operator for use with binary operations_old which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return The result of applying the specified binary operation to <code>src1</code> and <code>src2</code>
     * element-wise.
     * @throws IllegalArgumentException If <code>src1</code> and <code>src2</code> do not have the same shape.
     */
    public static CsrCMatrixOld applyBinOpp(CsrMatrixOld src1, CsrCMatrixOld src2,
                                            BiFunction<Double, CNumber, CNumber> opp,
                                            UnaryOperator<CNumber> uOpp) {
        ParameterChecks.ensureEqualShape(src1.shape, src2.shape);

        List<CNumber> dest = new ArrayList<>();
        int[] rowPointers = new int[src1.rowPointers.length];
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<src1.numRows; i++) {
            int rowPtr1 = src1.rowPointers[i];
            int rowPtr2 = src2.rowPointers[i];

            while(rowPtr1 < src1.rowPointers[i+1] && rowPtr2 < src2.rowPointers[i+1]) {
                int col1 = src1.colIndices[rowPtr1];
                int col2 = src2.colIndices[rowPtr2];

                if(col1 == col2) {
                    dest.add(opp.apply(src1.entries[rowPtr1], src2.entries[rowPtr2]));
                    colIndices.add(col1);
                    rowPtr1++;
                    rowPtr2++;
                } else if(col1 < col2) {
                    dest.add(new CNumber(src1.entries[rowPtr1]));
                    colIndices.add(col1);
                    rowPtr1++;
                } else {
                    if(uOpp!=null) dest.add(uOpp.apply(src2.entries[rowPtr2]));
                    else dest.add(src2.entries[rowPtr2]);
                    colIndices.add(col2);
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }

            while(rowPtr1 < src1.rowPointers[i+1]) {
                dest.add(new CNumber(src1.entries[rowPtr1]));
                colIndices.add(src1.colIndices[rowPtr1]);
                rowPtr1++;
                rowPointers[i+1]++;
            }

            while(rowPtr2 < src2.rowPointers[i+1]) {
                if(uOpp!=null) dest.add(uOpp.apply(src2.entries[rowPtr2]));
                else dest.add(src2.entries[rowPtr2]);
                colIndices.add(src2.colIndices[rowPtr2]);
                rowPtr2++;
                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrCMatrixOld(src1.shape,
                dest.toArray(CNumber[]::new),
                rowPointers,
                ArrayUtils.fromIntegerList(colIndices)
        );
    }


    /**
     * Applies an element-wise binary operation to two {@link CsrMatrixOld CSR Matrices}. <br><br>
     *
     * Note, this methods efficiency relies heavily on the assumption that both operand matrices are very large and very
     * sparse. If the two matrices are not large and very sparse, this method will likely be
     * significantly slower than simply converting the matrices to {@link MatrixOld dense matrices} and using a dense
     * matrix addition algorithm.
     * @param src1 The first matrix in the operation.
     * @param src2 The second matrix in the operation.
     * @param opp Binary operator to apply element-wise to <code>src1</code> and <code>src2</code>.
     * @param uOpp Unary operator for use with binary operations_old which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return The result of applying the specified binary operation to <code>src1</code> and <code>src2</code>
     * element-wise.
     * @throws IllegalArgumentException If <code>src1</code> and <code>src2</code> do not have the same shape.
     */
    public static CsrCMatrixOld applyBinOpp(CsrCMatrixOld src1, CsrMatrixOld src2,
                                            BiFunction<CNumber, Double, CNumber> opp,
                                            UnaryOperator<Double> uOpp) {
        ParameterChecks.ensureEqualShape(src1.shape, src2.shape);

        List<CNumber> dest = new ArrayList<>();
        int[] rowPointers = new int[src1.rowPointers.length];
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<src1.numRows; i++) {
            int rowPtr1 = src1.rowPointers[i];
            int rowPtr2 = src2.rowPointers[i];

            while(rowPtr1 < src1.rowPointers[i+1] && rowPtr2 < src2.rowPointers[i+1]) {
                int col1 = src1.colIndices[rowPtr1];
                int col2 = src2.colIndices[rowPtr2];

                if(col1 == col2) {
                    double val2 = uOpp==null ? src2.entries[rowPtr2] : uOpp.apply(src2.entries[rowPtr2]);
                    dest.add(opp.apply(src1.entries[rowPtr1], val2));
                    colIndices.add(col1);
                    rowPtr1++;
                    rowPtr2++;
                } else if(col1 < col2) {
                    dest.add(src1.entries[rowPtr1]);
                    colIndices.add(col1);
                    rowPtr1++;
                } else {
                    if(uOpp!=null) dest.add(new CNumber(uOpp.apply(src2.entries[rowPtr2])));
                    else dest.add(new CNumber(src2.entries[rowPtr2]));
                    colIndices.add(col2);
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }

            while(rowPtr1 < src1.rowPointers[i+1]) {
                dest.add(src1.entries[rowPtr1]);
                colIndices.add(src1.colIndices[rowPtr1]);
                rowPtr1++;
                rowPointers[i+1]++;
            }

            while(rowPtr2 < src2.rowPointers[i+1]) {
                if(uOpp!=null) dest.add(new CNumber(uOpp.apply(src2.entries[rowPtr2])));
                else dest.add(new CNumber(src2.entries[rowPtr2]));
                colIndices.add(src2.colIndices[rowPtr2]);
                rowPtr2++;
                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrCMatrixOld(src1.shape,
                dest.toArray(CNumber[]::new),
                rowPointers,
                ArrayUtils.fromIntegerList(colIndices)
        );
    }


     /**
     * Computes the element-wise multiplication between a complex sparse matrix and a real sparse matrix. <br><br>
     *
     * @param src1 The first matrix in the element-wise multiplication.
     * @param src2 The second matrix in the element-wise multiplication.
     * @return The result of the element-wise multiplication between <code>src1</code> and <code>src2</code>.
     * @throws IllegalArgumentException If <code>src1</code> and <code>src2</code> do not have the same shape.
     */
    public static CsrCMatrixOld elemMult(CsrCMatrixOld src1, CsrMatrixOld src2) {
        ParameterChecks.ensureEqualShape(src1.shape, src2.shape);

        List<CNumber> dest = new ArrayList<>();
        int[] rowPointers = new int[src1.rowPointers.length];
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<src1.numRows; i++) {
            int rowPtr1 = src1.rowPointers[i];
            int rowPtr2 = src2.rowPointers[i];

            while(rowPtr1 < src1.rowPointers[i+1] && rowPtr2 < src2.rowPointers[i+1]) {
                int col1 = src1.colIndices[rowPtr1];
                int col2 = src2.colIndices[rowPtr2];

                if(col1 == col2) { // Only values at the same indices need to be multiplied.
                    dest.add(src1.entries[rowPtr1].mult(src2.entries[rowPtr2]));
                    colIndices.add(col1);
                    rowPtr1++;
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrCMatrixOld(src1.shape,
                dest.toArray(CNumber[]::new),
                rowPointers,
                ArrayUtils.fromIntegerList(colIndices)
        );
    }
}
