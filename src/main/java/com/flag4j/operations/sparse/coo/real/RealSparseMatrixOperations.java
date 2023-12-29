package com.flag4j.operations.sparse.coo.real;

import com.flag4j.CooMatrix;
import com.flag4j.Matrix;
import com.flag4j.SparseVector;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * This class has low level implementations for operations between two real sparse matrices.
 */
public class RealSparseMatrixOperations {

    private RealSparseMatrixOperations() {
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
    public static CooMatrix add(CooMatrix src1, CooMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.entries.length, src2.entries.length);

        List<Double> sum = new ArrayList<>(initCapacity);
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
                sum.add(src1.entries[src1Counter] + src2.entries[src2Counter]);
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                sum.add(src1.entries[src1Counter]);
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
            } else {
                sum.add(src2.entries[src2Counter]);
                rowIndices.add(src2.rowIndices[src2Counter]);
                colIndices.add(src2.colIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new CooMatrix(
                src1.shape,
                sum,
                rowIndices,
                colIndices
        );
    }


    /**
     * Adds a double all entries (including zero values) of a real sparse matrix.
     * @param src Sparse matrix to add double value to.
     * @param a Double value to add to the sparse matrix.
     * @return The result of the matrix addition.
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} entries in the matrix (including zero entries).
     */
    public static Matrix add(CooMatrix src, double a) {
        double[] sum = new double[src.totalEntries().intValueExact()];
        Arrays.fill(sum, a);

        int row;
        int col;

        for(int i=0; i<src.entries.length; i++) {
            row = src.rowIndices[i];
            col = src.colIndices[i];
            sum[row*src.numCols + col] += src.entries[i];
        }

        return new Matrix(src.shape.copy(), sum);
    }


    /**
     * Computes the subtraction between two real sparse matrices. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the subtraction.
     * @param src2 Second matrix in the subtraction.
     * @return The difference of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static CooMatrix sub(CooMatrix src1, CooMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.entries.length, src2.entries.length);

        List<Double> sum = new ArrayList<>(initCapacity);
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
                sum.add(src1.entries[src1Counter] - src2.entries[src2Counter]);
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                sum.add(src1.entries[src1Counter]);
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
            } else {
                sum.add(-src2.entries[src2Counter]);
                rowIndices.add(src2.rowIndices[src2Counter]);
                colIndices.add(src2.colIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new CooMatrix(
                src1.shape,
                sum,
                rowIndices,
                colIndices
        );
    }


    /**
     * Subtracts a double from all entries (including zero values) of a real sparse matrix.
     * @param src Sparse matrix to subtract double value from.
     * @param a Double value to subtract from the sparse matrix.
     * @return The result of the matrix subtraction.
     * @throws ArithmeticException If the {@code src} sparse matrix is too large to be converted to a dense matrix.
     * That is, there are more than {@link Integer#MAX_VALUE} entries in the matrix (including zero entries).
     */
    public static Matrix sub(CooMatrix src, double a) {
        double[] sum = new double[src.totalEntries().intValueExact()];
        Arrays.fill(sum, -a);

        int row;
        int col;

        for(int i=0; i<src.entries.length; i++) {
            row = src.rowIndices[i];
            col = src.colIndices[i];
            sum[row*src.numCols + col] += src.entries[i];
        }

        return new Matrix(src.shape.copy(), sum);
    }



    /**
     * Multiplies two sparse matrices element-wise. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the element-wise multiplication.
     * @param src2 Second matrix in the element-wise multiplication.
     * @return The element-wise product of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static CooMatrix elemMult(CooMatrix src1, CooMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.entries.length, src2.entries.length);

        List<Double> product = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]
                    && src1.colIndices[src1Counter] == src2.colIndices[src2Counter]) {
                product.add(src1.entries[src1Counter]*src2.entries[src2Counter]);
                rowIndices.add(src1.rowIndices[src1Counter]);
                colIndices.add(src1.colIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(src1.rowIndices[src1Counter] == src2.rowIndices[src2Counter]) {
                // Matching row indices.

                if(src1.colIndices[src1Counter] < src2.colIndices[src2Counter]) {
                    src1Counter++;
                } else {
                    src2Counter++;
                }
            } else {
                if(src1.rowIndices[src1Counter] < src2.rowIndices[src2Counter]) {
                    src1Counter++;
                } else {
                    src2Counter++;
                }
            }
        }

        return new CooMatrix(
                src1.shape,
                product,
                rowIndices,
                colIndices
        );
    }


    /**
     * Adds a sparse vector to each column of a sparse matrix as if the vector is a column vector.
     * @param src The source sparse matrix.
     * @param col Sparse vector to add to each column of the sparse matrix.
     * @return A dense copy of the {@code src} matrix with the {@code col} vector added to each row of the matrix.
     */
    public static Matrix addToEachCol(CooMatrix src, SparseVector col) {
        ParameterChecks.assertEquals(src.numRows, col.size);
        double[] destEntries = new double[src.totalEntries().intValueExact()];

        // Add values from sparse matrix.
        for(int i=0; i<src.entries.length; i++) {
            destEntries[src.rowIndices[i]*src.numCols + src.colIndices[i]] = src.entries[i];
        }

        // Add values from sparse column.
        for(int i=0; i<col.entries.length; i++) {
            int idx = col.indices[i]*src.numCols;
            int end = idx + src.numCols;
            double value = col.entries[i];

            while(idx < end) {
                destEntries[idx++] += value;
            }
        }

        return new Matrix(src.shape.copy(), destEntries);
    }


    /**
     * Adds a sparse vector to each row of a sparse matrix as if the vector is a row vector.
     * @param src The source sparse matrix.
     * @param row Sparse vector to add to each row of the sparse matrix.
     * @return A dense copy of the {@code src} matrix with the {@code row} vector added to each row of the matrix.
     */
    public static Matrix addToEachRow(CooMatrix src, SparseVector row) {
        ParameterChecks.assertEquals(src.numCols, row.size);
        double[] destEntries = new double[src.totalEntries().intValueExact()];

        // Add values from sparse matrix.
        for(int i=0; i<src.entries.length; i++) {
            destEntries[src.rowIndices[i]*src.numCols + src.colIndices[i]] = src.entries[i];
        }

        // Add values from sparse column.
        for(int i=0; i<row.entries.length; i++) {
            int idx = 0;
            int colIdx = row.indices[i];
            double value = row.entries[i];

            while(idx < destEntries.length) {
                destEntries[idx + colIdx] += value;
                idx += src.numCols;
            }
        }

        return new Matrix(src.shape.copy(), destEntries);
    }
}
