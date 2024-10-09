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

package org.flag4j.linalg.operations.dense_sparse.coo.real_field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.CooFieldMatrixBase;
import org.flag4j.arrays.backend.DenseFieldMatrixBase;
import org.flag4j.arrays.backend.DenseFieldVectorBase;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.linalg.operations.common.field_ops.FieldOperations;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * This class contains low level implementations of operations between real/field and dense/sparse matrices.
 */
public final class RealFieldDenseCooMatrixOperations {

    private RealFieldDenseCooMatrixOperations() {
        // Hide private constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Adds a real dense matrix to a real sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of the matrix addition.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> DenseFieldMatrixBase<?, ?, ?, ?, T> add(
            DenseFieldMatrixBase<?, ?, ?, ?, T> src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldMatrixBase<?, ?, ?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.entries[idx] = dest.entries[idx].add(src2.entries[i]);
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
    public static <T extends Field<T>> DenseFieldMatrixBase<?, ?, ?, ?, T> sub(DenseFieldMatrixBase<?, ?, ?, ?, T> src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldMatrixBase<?, ?, ?, ?, T> dest = src1.copy();

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.entries[idx] = dest.entries[idx].sub(src2.entries[i]);
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
    public static <T extends Field<T>> DenseFieldMatrixBase<?, ?, ?, ?, T> sub(CooMatrix src2, DenseFieldMatrixBase<?, ?, ?, ?, T> src1) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        DenseFieldMatrixBase<?, ?, ?, ?, T> dest = src1.makeLikeTensor(src1.shape, FieldOperations.scalMult(src1.entries, -1));

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            dest.entries[idx] = dest.entries[idx].add(src2.entries[i]);
        }

        return dest;
    }


    /**
     * Adds a complex dense matrix to a real sparse matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, the storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> void addEq(DenseFieldMatrixBase<?, ?, ?, ?, T> src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            src1.entries[idx] = src1.entries[idx].add(src2.entries[i]);
        }
    }


    /**
     * Subtracts a real sparse matrix from a complex dense matrix and stores the result in the first matrix.
     * @param src1 Entries of first matrix in the sum. Also, the storage for the result.
     * @param src2 Entries of second matrix in the sum.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> void subEq(DenseFieldMatrixBase<?, ?, ?, ?, T> src1, CooMatrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        for(int i=0; i<src2.nnz; i++) {
            int idx = src2.rowIndices[i]*src1.numCols + src2.colIndices[i];
            src1.entries[idx] = src1.entries[idx].sub(src2.entries[i]);
        }
    }


    /**
     * Computes the element-wise multiplication between a real dense matrix and a complex sparse matrix.
     * @param src1 First matrix.
     * @param src2 Second matrix.
     * @return The result of element-wise multiplication.
     * @throws IllegalArgumentException If the matrices do not have the same shape.
     */
    public static <T extends Field<T>> CooFieldMatrixBase<?, ?, ?, ?, T> elemMult(
            Matrix src1, CooFieldMatrixBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Field<T>[] destEntries = new Field[src2.nnz];

        for(int i=0; i<destEntries.length; i++) {
            int row = src2.rowIndices[i];
            int col = src2.colIndices[i];
            destEntries[i] = src2.entries[i].mult(src1.entries[row*src1.numCols + col]);
        }

        return src2.makeLikeTensor(src2.shape, destEntries, src2.rowIndices.clone(), src2.colIndices.clone());
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
    public static <T extends Field<T>> CooFieldMatrixBase<?, ?, ?, ?, T> elemDiv(CooFieldMatrixBase<?, ?, ?, ?, T> src1, Matrix src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        Field<T>[] quotient = new Field[src1.entries.length];

        for(int i=0; i<src1.entries.length; i++) {
            int row = src1.rowIndices[i];
            int col = src1.colIndices[i];
            quotient[i] = src1.entries[i].div(src2.entries[row*src2.numCols + col]);
        }

        return src1.makeLikeTensor(src1.shape, quotient, src1.rowIndices.clone(), src1.colIndices.clone());
    }


    /**
     * Adds a dense vector to each column as if the vector is a column vector.
     * @param src Source sparse matrix.
     * @param col Vector to add to each column of the source matrix.
     * @return A dense copy of the {@code src} matrix with the specified vector added to each column.
     * @throws IllegalArgumentException If the number of entries in the {@code col} vector does not match the number
     * of rows in the {@code src} matrix.
     */
    public static <T extends Field<T>> DenseFieldMatrixBase<?, ?, ?, ?, T> addToEachCol(
            CooMatrix src, DenseFieldVectorBase<?, ?, ?, T> col) {
        Field<T>[] sumEntries = new Field[src.shape.totalEntriesIntValueExact()];
        Arrays.fill(sumEntries, col.getZeroElement());
        DenseFieldMatrixBase<?, ?, ?, ?, T> sum = col.makeLikeMatrix(src.shape, sumEntries);

        for(int j=0; j<sum.numCols; j++)
            sum.setCol(col.entries, j);

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.rowIndices[i]*src.numCols + src.colIndices[i];
            sum.entries[idx] = sum.entries[idx].add(src.entries[i]);
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
    public static <T extends Field<T>> DenseFieldMatrixBase<?, ?, ?, ?, T> addToEachRow(
            CooMatrix src, DenseFieldVectorBase<?, ?, ?, T> row) {
        Field<T>[] sumEntries = new Field[src.shape.totalEntriesIntValueExact()];
        Arrays.fill(sumEntries, row.getZeroElement());
        DenseFieldMatrixBase<?, ?, ?, ?, T> sum = row.makeLikeMatrix(src.shape, sumEntries);

        for(int i=0; i<sum.numRows; i++)
            sum.setRow(row.entries, i);

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.rowIndices[i]*src.numCols + src.colIndices[i];
            sum.entries[idx] = sum.entries[idx].add(src.entries[i]);
        }

        return sum;
    }
}
