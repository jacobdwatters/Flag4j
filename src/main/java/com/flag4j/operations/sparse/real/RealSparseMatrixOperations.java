package com.flag4j.operations.sparse.real;

import com.flag4j.Matrix;
import com.flag4j.SparseMatrix;
import com.flag4j.Vector;
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
    public static SparseMatrix add(SparseMatrix src1, SparseMatrix src2) {
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

        return new SparseMatrix(
                src1.shape,
                sum.stream().mapToDouble(Double::doubleValue).toArray(),
                rowIndices.stream().mapToInt(Integer::intValue).toArray(),
                colIndices.stream().mapToInt(Integer::intValue).toArray()
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
    public static Matrix add(SparseMatrix src, double a) {
        double[] sum = new double[src.totalEntries().intValueExact()];
        Arrays.fill(sum, a);

        int row;
        int col;

        for(int i=0; i<src.entries.length; i++) {
            row = src.rowIndices[i];
            col = src.colIndices[i];
            sum[row*src.numCols + col] = src.entries[i];
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
    public static SparseMatrix sub(SparseMatrix src1, SparseMatrix src2) {
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

        return new SparseMatrix(
                src1.shape,
                sum.stream().mapToDouble(Double::doubleValue).toArray(),
                rowIndices.stream().mapToInt(Integer::intValue).toArray(),
                colIndices.stream().mapToInt(Integer::intValue).toArray()
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
    public static Matrix sub(SparseMatrix src, double a) {
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
     * Multiplies two sparse matrices element-wise. This method assumes that the indices of the two matrices are sorted
     * lexicographically.
     * @param src1 First matrix in the element-wise multiplication.
     * @param src2 Second matrix in the element-wise multiplication.
     * @return The element-wise product of the two matrices {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static SparseMatrix elemMult(SparseMatrix src1, SparseMatrix src2) {
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

        return new SparseMatrix(
                src1.shape,
                product.stream().mapToDouble(Double::doubleValue).toArray(),
                rowIndices.stream().mapToInt(Integer::intValue).toArray(),
                colIndices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Adds a dense vector to each column as if the vector is a column vector.
     * @param src Source sparse matrix.
     * @param col Vector to add to each column of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each column.
     * @throws IllegalArgumentException If the number of entries in the {@code col} vector does not match the number
     * of rows in the {@code src} matrix.
     */
    public static Matrix addToEachCol(SparseMatrix src, Vector col) {
        Matrix sum = new Matrix(src.numRows, src.numCols);

        for(int j=0; j<sum.numCols; j++) {
            sum.setCol(col, j);
        }

        for(int i=0; i<src.entries.length; i++) {
            sum.entries[src.rowIndices[i]*src.numCols + src.colIndices[i]] += src.entries[i];
        }

        return sum;
    }


    /**
     * Adds a dense vector to add to each row as if the vector is a row vector.
     * @param src Source sparse matrix.
     * @param row Vector to add to each row of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each row.
     * @throws IllegalArgumentException If the number of entries in the {@code col} vector does not match the number
     * of columns in the {@code src} matrix.
     */
    public static Matrix addToEachRow(SparseMatrix src, Vector row) {
        Matrix sum = new Matrix(src.numRows, src.numCols);

        for(int i=0; i<sum.numRows; i++) {
            sum.setRow(row.entries, i);
        }

        for(int i=0; i<src.entries.length; i++) {
            sum.entries[src.rowIndices[i]*src.numCols + src.colIndices[i]] += src.entries[i];
        }

        return sum;
    }
}
