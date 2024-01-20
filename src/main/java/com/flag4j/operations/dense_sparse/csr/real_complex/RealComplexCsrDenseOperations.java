package com.flag4j.operations.dense_sparse.csr.real_complex;

import com.flag4j.CMatrix;
import com.flag4j.CsrCMatrix;
import com.flag4j.CsrMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

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
}
