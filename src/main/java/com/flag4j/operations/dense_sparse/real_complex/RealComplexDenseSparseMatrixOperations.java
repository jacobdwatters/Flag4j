package com.flag4j.operations.dense_sparse.real_complex;


import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

/**
 * This class contains low level implementations of operations between real/complex and dense/sparse matrices.
 */
public class RealComplexDenseSparseMatrixOperations {

    private RealComplexDenseSparseMatrixOperations() {
        // Hide private constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Adds a real dense matrix to a complex sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrix add(Matrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrix dest = new CMatrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col].addEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Adds a real dense matrix to a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrix add(CMatrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrix dest = new CMatrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col].addEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrix sub(Matrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrix dest = new CMatrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col].subEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a complex dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrix sub(CMatrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrix dest = new CMatrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col].subEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrix sub(SparseCMatrix src2, Matrix src1) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrix dest = new CMatrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col] = dest.entries[row*src1.numCols + col].addInv();
            dest.entries[row*src1.numCols + col].subEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Subtracts a real sparse matrix from a real dense matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix subtraction.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static CMatrix sub(SparseMatrix src2, CMatrix src1) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CMatrix dest = new CMatrix(src1);

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            dest.entries[row*src1.numCols + col] = dest.entries[row*src1.numCols + col].addInv();
            dest.entries[row*src1.numCols + col].subEq(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Adds a complex dense matrix to a real sparse matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, the storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void addEq(CMatrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.entries[row*src1.numCols + col].addEq(src2.entries[i]);
        }
    }


    /**
     * Subtracts a real sparse matrix from a complex dense matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, the storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static void subEq(CMatrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;

        for(int i=0; i<src2.nonZeroEntries(); i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            src1.entries[row*src1.numCols + col].subEq(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise multiplication between a real dense matrix and a complex sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static SparseCMatrix elemMult(Matrix src1, SparseCMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CNumber[] destEntries = new CNumber[src2.nonZeroEntries()];

        for(int i=0; i<destEntries.length; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            destEntries[i] = src2.entries[i].mult(src1.entries[row*src1.numCols + col]);
        }

        return new SparseCMatrix(src2.shape.copy(), destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise multiplication between a complex dense matrix and a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static SparseCMatrix elemMult(CMatrix src1, SparseMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        int row, col;
        CNumber[] destEntries = new CNumber[src2.nonZeroEntries()];

        for(int i=0; i<destEntries.length; i++) {
            row = src2.rowIndices[i];
            col = src2.colIndices[i];
            destEntries[i] = src1.entries[row*src1.numCols + col].mult(src2.entries[i]);
        }

        return new SparseCMatrix(src2.shape.copy(), destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
    }


    /**
     * Computes the element-wise division between a complex sparse matrix and a real dense matrix.
     *
     * <p>
     *     If the dense matrix contains a zero at the same index the sparse matrix contains a non-zero, the result will be
     *     either {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * </p>
     *
     * <p>
     *     If the dense matrix contains a zero at an index for which the sparse matrix is also zero, the result will be
     *     zero. This is done to realize computational benefits from operations with sparse matrices.
     * </p>
     *
     * @param src1 Real sparse matrix and numerator in element-wise quotient.
     * @param src2 Real Dense matrix and denominator in element-wise quotient.
     * @return The element-wise quotient of {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static SparseCMatrix elemDiv(SparseCMatrix src1, Matrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] quotient = new CNumber[src1.entries.length];

        int row;
        int col;

        for(int i=0; i<src1.entries.length; i++) {
            row = src1.rowIndices[i];
            col = src1.colIndices[i];
            quotient[i] = src1.entries[i].div(src2.entries[row*src2.numCols + col]);
        }

        return new SparseCMatrix(src1.shape.copy(), quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }


    /**
     * Computes the element-wise division between a real sparse matrix and a complex dense matrix.
     *
     * <p>
     *     If the dense matrix contains a zero at the same index the sparse matrix contains a non-zero, the result will be
     *     either {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * </p>
     *
     * <p>
     *     If the dense matrix contains a zero at an index for which the sparse matrix is also zero, the result will be
     *     zero. This is done to realize computational benefits from operations with sparse matrices.
     * </p>
     *
     * @param src1 Real sparse matrix and numerator in element-wise quotient.
     * @param src2 Real Dense matrix and denominator in element-wise quotient.
     * @return The element-wise quotient of {@code src1} and {@code src2}.
     * @throws IllegalArgumentException If {@code src1} and {@code src2} do not have the same shape.
     */
    public static SparseCMatrix elemDiv(SparseMatrix src1, CMatrix src2) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        CNumber[] quotient = new CNumber[src1.entries.length];

        int row;
        int col;

        for(int i=0; i<src1.entries.length; i++) {
            row = src1.rowIndices[i];
            col = src1.colIndices[i];
            quotient[i] = new CNumber(src1.entries[i]).div(src2.entries[row*src2.numCols + col]);
        }

        return new SparseCMatrix(src1.shape.copy(), quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }


    /**
     * Adds a dense vector to each column as if the vector is a column vector.
     * @param src Source sparse matrix.
     * @param col Vector to add to each column of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each column.
     * @throws IllegalArgumentException If the number of entries in the {@code col} vector does not match the number
     * of rows in the {@code src} matrix.
     */
    public static CMatrix addToEachCol(SparseMatrix src, CVector col) {
        CMatrix sum = new CMatrix(src.numRows, src.numCols);

        for(int j=0; j<sum.numCols; j++) {
            sum.setCol(col, j);
        }

        for(int i=0; i<src.entries.length; i++) {
            sum.entries[src.rowIndices[i]*src.numCols + src.colIndices[i]].addEq(src.entries[i]);
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
    public static CMatrix addToEachRow(SparseMatrix src, CVector row) {
        CMatrix sum = new CMatrix(src.numRows, src.numCols);

        for(int i=0; i<sum.numRows; i++) {
            sum.setRow(row.entries, i);
        }

        for(int i=0; i<src.entries.length; i++) {
            sum.entries[src.rowIndices[i]*src.numCols + src.colIndices[i]].addEq(src.entries[i]);
        }

        return sum;
    }


    /**
     * Adds a dense vector to each column as if the vector is a column vector.
     * @param src Source sparse matrix.
     * @param col Vector to add to each column of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each column.
     * @throws IllegalArgumentException If the number of entries in the {@code col} vector does not match the number
     * of rows in the {@code src} matrix.
     */
    public static CMatrix addToEachCol(SparseCMatrix src, Vector col) {
        CMatrix sum = new CMatrix(src.numRows, src.numCols);

        for(int j=0; j<sum.numCols; j++) {
            sum.setCol(col.entries, j);
        }

        for(int i=0; i<src.entries.length; i++) {
            sum.entries[src.rowIndices[i]*src.numCols + src.colIndices[i]].addEq(src.entries[i]);
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
    public static CMatrix addToEachRow(SparseCMatrix src, Vector row) {
        CMatrix sum = new CMatrix(src.numRows, src.numCols);

        for(int i=0; i<sum.numRows; i++) {
            sum.setRow(row.entries, i);
        }

        for(int i=0; i<src.entries.length; i++) {
            sum.entries[src.rowIndices[i]*src.numCols + src.colIndices[i]].addEq(src.entries[i]);
        }

        return sum;
    }
}
