package com.flag4j.operations.sparse.csr.real;

import com.flag4j.CsrMatrix;
import com.flag4j.operations.common.real.RealProperties;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class contains methods to check equality or approximate equality between two sparse CSR matrices.
 */
public final class RealCsrEquals {

    private RealCsrEquals() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     * @param src1 First matrix in the comparison.
     * @param src2 Second matrix in the comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if the {@code src1} matrix is the same shape as the {@code src2} matrix and all entries
     * are 'close', i.e. elements {@code a} and {@code b} at the same positions in the two matrices respectively
     * satisfy {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns false.
     */
    public static boolean allClose(CsrMatrix src1, CsrMatrix src2, double relTol, double absTol) {
        boolean close = src1.shape.equals(src2.shape);

        if(close) {
            // Remove values which are 'close' to zero.
            List<Double> src1Entries = new ArrayList<>(src1.entries.length);
            List<Integer> src1ColIndices = new ArrayList<>(src1Entries.size());
            int[] src1RowPointers = new int[src1.rowPointers.length];
            removeCloseToZero(src1, src1Entries, src1RowPointers, src1ColIndices, absTol);

            List<Double> src2Entries = new ArrayList<>(src2.entries.length);
            List<Integer> src2ColIndices = new ArrayList<>(src2Entries.size());
            int[] src2RowPointers = new int[src2.rowPointers.length];
            removeCloseToZero(src2, src2Entries, src2RowPointers, src2ColIndices, absTol);

            close = Arrays.equals(src1RowPointers, src2RowPointers)

                    && Arrays.equals(ArrayUtils.fromIntegerList(src1ColIndices),
                    ArrayUtils.fromIntegerList(src2ColIndices))

                    && RealProperties.allClose(ArrayUtils.fromDoubleList(src1Entries),
                    ArrayUtils.fromDoubleList(src2Entries), relTol, absTol);
        }

        return close;
    }


    /**
     * Removes entries in {@code src} which are within {@code atol} in absolute value from zero.
     * @param src Source CSR matrix.
     * @param entries List to store value in {@code src} which are not within {@code atol} in absolute value from zero.
     * @param colIndices Column indices of entries.
     * @param rowPointers Row pointers for entries.
     * @param aTol Absolute tolerance for value to be considered close to zero.
     */
    private static void removeCloseToZero(CsrMatrix src, List<Double> entries, int[] rowPointers,
                                          List<Integer> colIndices, double aTol) {
        for(int i=0; i<src.numRows; i++) {
            int start = src.rowPointers[i];
            int stop = src.rowPointers[i+1];

            for(int j=start; j<stop; j++) {
                double value = src.entries[j];

                if(Math.abs(value) > aTol) {
                    // Then keep value.
                    entries.add(value);
                    colIndices.add(src.colIndices[j]);
                    rowPointers[i]++;
                }
            }
        }

        // Accumulate row pointers.
        int size = rowPointers.length-1;
        for(int i=0; i<size; i++) {
            rowPointers[i+1] += rowPointers[i];
        }
    }
}
