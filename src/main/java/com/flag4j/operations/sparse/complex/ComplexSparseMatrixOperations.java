package com.flag4j.operations.sparse.complex;

import com.flag4j.CMatrix;
import com.flag4j.SparseCMatrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.List;

/**
 * This class has low level implementations for operations between two complex sparse matrices.
 */
public class ComplexSparseMatrixOperations {

    private ComplexSparseMatrixOperations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds two real sparse matrices. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the sum.
     * @param src2 Second matrix in the sum.
     * @return The sum of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static SparseCMatrix add(SparseCMatrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.entries.length, src2.entries.length);

        List<CNumber> values = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1.entries.length || src2Counter < src2.entries.length) {

            if(src1Counter >= src1.entries.length || src2Counter >= src2.entries.length) {
                add1 = src2Counter >= src2.entries.length;
                add2 = !add1;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1.colIndices[src1Counter] < src2.colIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                values.add(src1.entries[src1Counter].add(src2.entries[src2Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                values.add(src1.entries[src1Counter].copy());
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2.entries[src2Counter].copy());
                rowIndices.add(src2.rowIndices[src2Counter]);
                colIndices.add(src2.colIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new SparseCMatrix(
                src1.shape,
                values.toArray(CNumber[]::new),
                rowIndices.stream().mapToInt(Integer::intValue).toArray(),
                colIndices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Adds a complex number to all entries (including zero values) of a complex sparse matrix.
     * @param src Sparse matrix to add complex value to.
     * @param a Complex value to add to the sparse matrix.
     * @return The result of the matrix addition.
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} entries in the matrix (including zero entries).
     */
    public static CMatrix add(SparseCMatrix src, CNumber a) {
        CNumber[] sum = new CNumber[src.totalEntries().intValueExact()];
        ArrayUtils.fill(sum, a);

        int row;
        int col;

        for(int i=0; i<src.entries.length; i++) {
            row = src.rowIndices[i];
            col = src.colIndices[i];
            sum[row*src.numCols + col].addEq(src.entries[i]);
        }

        return new CMatrix(src.shape.copy(), sum);
    }


    /**
     * Computes the difference between two complex matrices. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the difference.
     * @param src2 Second matrix in the difference.
     * @return The difference of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static SparseCMatrix sub(SparseCMatrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.entries.length, src2.entries.length);

        List<CNumber> values = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1.entries.length || src2Counter < src2.entries.length) {

            if(src1Counter >= src1.entries.length || src2Counter >= src2.entries.length) {
                add1 = src2Counter >= src2.entries.length;
                add2 = !add1;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1.colIndices[src1Counter] < src2.colIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                values.add(src1.entries[src1Counter].sub(src2.entries[src2Counter]));
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                values.add(src1.entries[src1Counter].copy());
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2.entries[src2Counter].addInv());
                rowIndices.add(src2.rowIndices[src2Counter]);
                colIndices.add(src2.colIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new SparseCMatrix(
                src1.shape,
                values.toArray(CNumber[]::new),
                rowIndices.stream().mapToInt(Integer::intValue).toArray(),
                colIndices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Subtracts a complex number from all entries (including zero values) of a complex sparse matrix.
     * @param src Sparse matrix to subtract complex value from.
     * @param a Complex value to add to the sparse matrix.
     * @return The result of the matrix addition.
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} entries in the matrix (including zero entries).
     */
    public static CMatrix sub(SparseCMatrix src, CNumber a) {
        return add(src, a.addInv());
    }
}
