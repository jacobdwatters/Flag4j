package com.flag4j.operations.sparse.csr.real_complex;

import com.flag4j.CsrCMatrix;
import com.flag4j.CsrMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/**
 * This class contains low-level implementations for element-wise operations on real/complex CSR matrices.
 */
public class RealComplexCsrOperations {

    private RealComplexCsrOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Applies an element-wise binary operation to two {@link CsrMatrix CSR Matrices}. <br><br>
     *
     * Note, this methods efficiency relies heavily on the assumption that both operand matrices are very large and very
     * sparse. If the two matrices are not large and very sparse, this method will likely be
     * significantly slower than simply converting the matrices to {@link Matrix dense matrices} and using a dense
     * matrix addition algorithm.
     * @param src1 The first matrix in the operation.
     * @param src2 The second matrix in the operation.
     * @param opp Binary operator to apply element-wise to <code>src1</code> and <code>src2</code>.
     * @return The result of applying the specified binary operation to <code>src1</code> and <code>src2</code>
     * element-wise.
     * @throws IllegalArgumentException If <code>src1</code> and <code>src2</code> do not have the same shape.
     */
    public static CsrCMatrix applyBinOpp(CsrMatrix src1, CsrCMatrix src2, BiFunction<Double, CNumber, CNumber> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

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
                    dest.add(src2.entries[rowPtr2].copy());
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
                dest.add(src2.entries[rowPtr2].copy());
                colIndices.add(src2.colIndices[rowPtr2]);
                rowPtr2++;
                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrCMatrix(src1.shape.copy(),
                dest.toArray(CNumber[]::new),
                rowPointers,
                ArrayUtils.fromIntegerList(colIndices)
        );
    }


    /**
     * Applies an element-wise binary operation to two {@link CsrMatrix CSR Matrices}. <br><br>
     *
     * Note, this methods efficiency relies heavily on the assumption that both operand matrices are very large and very
     * sparse. If the two matrices are not large and very sparse, this method will likely be
     * significantly slower than simply converting the matrices to {@link Matrix dense matrices} and using a dense
     * matrix addition algorithm.
     * @param src1 The first matrix in the operation.
     * @param src2 The second matrix in the operation.
     * @param opp Binary operator to apply element-wise to <code>src1</code> and <code>src2</code>.
     * @return The result of applying the specified binary operation to <code>src1</code> and <code>src2</code>
     * element-wise.
     * @throws IllegalArgumentException If <code>src1</code> and <code>src2</code> do not have the same shape.
     */
    public static CsrCMatrix applyBinOpp(CsrCMatrix src1, CsrMatrix src2, BiFunction<CNumber, Double, CNumber> opp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

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
                    dest.add(src1.entries[rowPtr1].copy());
                    colIndices.add(col1);
                    rowPtr1++;
                } else {
                    dest.add(new CNumber(src2.entries[rowPtr2]));
                    colIndices.add(col2);
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }

            while(rowPtr1 < src1.rowPointers[i+1]) {
                dest.add(src1.entries[rowPtr1].copy());
                colIndices.add(src1.colIndices[rowPtr1]);
                rowPtr1++;
                rowPointers[i+1]++;
            }

            while(rowPtr2 < src2.rowPointers[i+1]) {
                dest.add(new CNumber(src2.entries[rowPtr2]));
                colIndices.add(src2.colIndices[rowPtr2]);
                rowPtr2++;
                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrCMatrix(src1.shape.copy(),
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
    public static CsrCMatrix elemMult(CsrCMatrix src1, CsrMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

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

        return new CsrCMatrix(src1.shape.copy(),
                dest.toArray(CNumber[]::new),
                rowPointers,
                ArrayUtils.fromIntegerList(colIndices)
        );
    }
}
