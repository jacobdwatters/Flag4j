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

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.function.BiFunction;

/**
 * This class contains low-level operations which act on a real/complex dense and a complex/real
 * sparse {@link CsrCMatrix CSR matrix}.
 */
public class RealComplexCsrDenseOperations {

    private RealComplexCsrDenseOperations() {
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
    public static CMatrix applyBinOpp(CsrCMatrix src1, Matrix src2, BiFunction<CNumber, Double, CNumber> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.
        CNumber[] dest = new CNumber[src2.entries.length];

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

        return new CMatrix(src2.shape.copy(), dest);
    }


    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrix applyBinOpp(Matrix src1, CsrCMatrix src2, BiFunction<Double, CNumber, CNumber> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.
        CNumber[] dest = new CNumber[src2.entries.length];

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

        return new CMatrix(src2.shape.copy(), dest);
    }


    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrix applyBinOpp(CsrMatrix src1, CMatrix src2, BiFunction<Double, CNumber, CNumber> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.
        CNumber[] dest = new CNumber[src2.entries.length];

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

        return new CMatrix(src2.shape.copy(), dest);
    }


    /**
     * Applies the specified binary operator element-wise to the two matrices.
     * @param src1 First matrix in element-wise binary operation.
     * @param src2 Second matrix in element-wise binary operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrix applyBinOpp(CMatrix src1, CsrMatrix src2, BiFunction<CNumber, Double, CNumber> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.
        CNumber[] dest = new CNumber[src2.entries.length];

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

        return new CMatrix(src2.shape.copy(), dest);
    }


    /**
     * Applies an element-wise binary operation to a real dense and real sparse CSR matrix under the
     * assumption that {@code opp.apply(x, 0d) = 0d} and {@code opp.apply(0d, x) = 0d}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static CsrCMatrix applyBinOppToSparse(Matrix src1, CsrCMatrix src2, BiFunction<Double, CNumber, CNumber> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

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

        return new CsrCMatrix(src1.shape.copy(), entries, rowPointers, colIndices);
    }


    /**
     * Applies an element-wise binary operation to a real dense and real sparse CSR matrix under the
     * assumption that {@code opp.apply(x, 0d) = 0d} and {@code opp.apply(0d, x) = 0d}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static CsrCMatrix applyBinOppToSparse(CMatrix src1, CsrMatrix src2, BiFunction<CNumber, Double, CNumber> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

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

        return new CsrCMatrix(src1.shape.copy(), entries, rowPointers, colIndices);
    }


    /**
     * Applies an element-wise binary operation to a real sparse and complex dense CSR matrix under the
     * assumption that {@code opp.apply(0d, x) = 0d} where {@code x} is a {@link CNumber}.
     * @param src1 The first matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param opp Operation to apply to the matrices.
     * @return The result of applying the operation element-wise to the matrices. Result is a sparse CSR matrix.
     */
    public static CsrCMatrix applyBinOppToSparse(CsrMatrix src1, CMatrix src2, BiFunction<Double, CNumber, CNumber> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape); // Ensure both matrices are same shape.

        int[] rowPointers = src1.rowPointers.clone();
        int[] colIndices = src1.colIndices.clone();
        CNumber[] entries = new CNumber[src1.entries.length];

        for(int i=0; i<src1.numRows; i++) {
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i+1];
            int src2RowOffset = i*src2.numCols;

            for(int j=start; j<stop; j++) {
                entries[j] = opp.apply(src1.entries[j], src2.entries[src2RowOffset + src1.colIndices[j]]);
            }
        }

        return new CsrCMatrix(src1.shape.copy(), entries, rowPointers, colIndices);
    }


    /**
     * Applies the specified binary operator element-wise to a matrix and a scalar.
     * @param src1 First matrix in element-wise binary operation.
     * @param b Scalar to apply elementwise using the specified operation.
     * @param opp Binary operator to apply element-wise to the two matrices.
     * @return A matrix containing the result from applying {@code opp} element-wise to the two matrices.
     */
    public static CMatrix applyBinOpp(CsrMatrix src1, CNumber b, BiFunction<Double, CNumber, CNumber> opp) {
        CNumber[] dest = new CNumber[src1.entries.length];
        ArrayUtils.fillZeros(dest);

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

        return new CMatrix(src1.shape.copy(), dest);
    }
}
