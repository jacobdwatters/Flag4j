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

package com.flag4j.operations.dense_sparse.csr.real;


import com.flag4j.dense.Matrix;
import com.flag4j.sparse.CsrMatrix;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.function.BinaryOperator;

/**
 * This class contains low-level operations which act on a real dense and a real sparse {@link CsrMatrix CSR matrix}.
 */
public class RealCsrDenseOperations {

    private RealCsrDenseOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static Matrix applyBinOpp(CsrMatrix src1, Matrix src2, BinaryOperator<Double> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.
        double[] dest = new double[src2.entries.length];

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[i];

                dest[idx] = opp.apply(
                        src1.entries[j],
                        src2.entries[idx]
                );
            }
        }

        return new Matrix(src2.shape.copy(), dest);
    }



    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static Matrix applyBinOpp(Matrix src1, CsrMatrix src2, BinaryOperator<Double> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.
        double[] dest = new double[src2.entries.length];

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

        return new Matrix(src2.shape.copy(), dest);
    }


    /**
     * Applies an element-wise binary operation to a real dense and real sparse CSR matrix under the
     * assumption that {@code opp.apply(x, 0d) = 0d} and {@code opp.apply(0d, x) = 0d}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static CsrMatrix applyBinOppToSparse(Matrix src1, CsrMatrix src2, BinaryOperator<Double> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        int[] rowPointers = src2.rowPointers.clone();
        int[] colIndices = src2.colIndices.clone();
        double[] entries = new double[src2.entries.length];


        for(int i=0; i<src2.rowPointers.length-1; i++) {
            int start = src2.rowPointers[i];
            int stop = src2.rowPointers[i+1];
            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src2.colIndices[j];

                entries[idx] = opp.apply(src1.entries[idx], src2.entries[j]);
            }
        }

        return new CsrMatrix(src1.shape.copy(), entries, rowPointers, colIndices);
    }


    /**
     * Applies the specified binary operator element-wise to a matrix and a scalar.
     * @param src1 First matrix in element-wise binary operation.
     * @param b Scalar to apply elementwise using the specified operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static Matrix applyBinOpp(CsrMatrix src1, double b, BinaryOperator<Double> opp) {
        double[] dest = new double[src1.entries.length];

        for(int i=0; i<src1.rowPointers.length-1; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];

            int rowOffset = i*src1.numCols;

            for(int j=start; j<stop; j++) {
                int idx = rowOffset + src1.colIndices[i];

                dest[idx] = opp.apply(
                        src1.entries[j],
                        b
                );
            }
        }

        return new Matrix(src1.shape.copy(), dest);
    }
}
