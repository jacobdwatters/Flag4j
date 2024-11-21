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

package org.flag4j.linalg.operations.sparse.csr.field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.CsrFieldMatrixBase;
import org.flag4j.arrays.backend_new.SparseMatrixData;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.*;

/**
 * This class contains low-level implementations of sparse-sparse {@link Field}
 * matrix multiplication where the sparse matrices are in CSR format.
 */
public final class CsrFieldMatMult {

    private CsrFieldMatMult() {
        // Hide default constructor for utility method.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * <p>Computes the matrix multiplication between two sparse CSR matrices.
     * <p>This method produces a dense matrix. If there is high confidence that the result will be sparse,
     * {@link #standardAsSparse(Shape, Field[], int[], int[], Shape, Field[], int[], int[])} is preferred.
     *
     * @param shape1 Shape of the first CSR matrix.
     * @param src1Entries Non-zero entries of the first CSR matrix.
     * @param src1RowPointers Non-zero row pointers of the first CSR matrix.
     * @param src1ColIndices Non-zero column indices of the first CSR matrix.
     * @param shape2 Shape of the second CSR matrix.
     * @param src2Entries Non-zero entries of the second CSR matrix.
     * @param src2RowPointers Non-zero row pointers of the second CSR matrix.
     * @param src2ColIndices Non-zero column indices of the second CSR matrix.
     * @return Entries of the dense matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static <T extends Field<T>> void standard(
            Shape shape1, Field<T>[] src1Entries, int[] src1RowPointers, int[] src1ColIndices,
            Shape shape2, Field<T>[] src2Entries, int[] src2RowPointers, int[] src2ColIndices,
            Field<T>[] dest) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        ValidateParameters.ensureMatMultShapes(shape1, shape2);

        Field<T> zero = null;
        if(src1Entries.length > 0) zero = src1Entries[0].getZero();
        else if(src2Entries.length > 0) zero = src2Entries[0].getZero();

        Arrays.fill(dest, zero);

        for(int i=0; i<rows1; i++) {
            int rowOffset = i*cols2;
            int stop = src1RowPointers[i+1];

            for(int aIndex=src1RowPointers[i]; aIndex<stop; aIndex++) {
                int aCol = src1ColIndices[aIndex];
                T aVal = (T) src1Entries[aIndex];
                int innerStop = src2RowPointers[aCol+1];

                for(int bIndex=src2RowPointers[aCol]; bIndex<innerStop; bIndex++) {
                    int bCol = src2ColIndices[bIndex];
                    Field<T> bVal = src2Entries[bIndex];

                    dest[rowOffset + bCol] = dest[rowOffset + bCol].add(bVal.mult(aVal));
                }
            }
        }
    }


    /**
     * <p>Computes the matrix multiplication between two sparse CSR matrices and returns the result as a sparse matrix.
     *
     * <p>Warning: This method <i>may</i> be slower than {@link #standard(CsrFieldMatrixBase, CsrFieldMatrixBase)}
     * if the result of multiplying this matrix with {@code src2} is not very sparse. Further, multiplying two
     * sparse matrices (even very sparse matrices) may result in a dense matrix so this method should be used with
     * caution. However, if there is confidence that the result will be sparse, this method should be preferred.
     *
     * @param shape1 Shape of the first CSR matrix.
     * @param src1Entries Non-zero entries of the first CSR matrix.
     * @param src1RowPointers Non-zero row pointers of the first CSR matrix.
     * @param src1ColIndices Non-zero column indices of the first CSR matrix.
     * @param shape2 Shape of the second CSR matrix.
     * @param src2Entries Non-zero entries of the second CSR matrix.
     * @param src2RowPointers Non-zero row pointers of the second CSR matrix.
     * @param src2ColIndices Non-zero column indices of the second CSR matrix.
     *
     * @return {@link SparseMatrixData Sparse matrix data object} containing the result of the matrix multiplication.
     *
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code shape1.get(1) != shape2.get(0)}.
     */
    public static <T extends Field<T>> SparseMatrixData<Field<T>> standardAsSparse(
            Shape shape1, Field<T>[] src1Entries, int[] src1RowPointers, int[] src1ColIndices,
            Shape shape2, Field<T>[] src2Entries, int[] src2RowPointers, int[] src2ColIndices) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(shape1, shape2);
        int src1NumRows = shape1.get(0);

        int[] resultRowPtr = new int[src1NumRows + 1];
        List<Field<T>> resultList = new ArrayList<>();
        List<Integer> resultColIndexList = new ArrayList<>();

        for (int i=0; i<src1NumRows; i++) {
            Map<Integer, T> tempMap = new HashMap<>();
            int start = src1RowPointers[i];
            int stop = src1RowPointers[i + 1];

            for (int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1ColIndices[aIndex];
                T aVal = (T) src1Entries[aIndex];
                int innerStart = src2RowPointers[aCol];
                int innerStop = src2RowPointers[aCol + 1];

                for (int bIndex=innerStart; bIndex<innerStop; bIndex++) {
                    int bCol = src2ColIndices[bIndex];
                    Field<T> bVal = src2Entries[bIndex];

                    tempMap.merge(bCol, bVal.mult(aVal), T::add);
                }
            }

            // Ensure entries within each row are sorted by the column indices.
            List<Integer> tempColIndices = new ArrayList<>(tempMap.keySet());
            Collections.sort(tempColIndices);

            for (int colIndex : tempColIndices) {
                resultColIndexList.add(colIndex);
                resultList.add(tempMap.get(colIndex));
            }

            resultRowPtr[i + 1] = resultList.size();
        }

        // TODO: Implement a CsrMatrixData<T> record to avoid having to copy the array to a list
        //  then re-copy it back to an array in the matrix constructor.
        List<Integer> rowPtrList = ArrayUtils.toArrayList(resultRowPtr);

        return new SparseMatrixData<>(new Shape(src1NumRows, shape2.get(1)),
                resultList, rowPtrList, resultColIndexList);
    }


    /**
     * Computes the matrix-vector multiplication between a sparse CSR matrix and a sparse COO vector.
     * @param shape Shape of the CSR matrix.
     * @param src1 Non-zero entries of the CSR matrix
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @param size Full size of the COO vector.
     * @param src2 Non-zero entries of the COO vector.
     * @param indices Non-zero indices of the COO Vector.
     * @param dest Array to store the dense result of the matrix-vector multiplication in.
     *
     * @throws IllegalArgumentException If {@code shape.get(1) != size} or {@code dest.length != size}.
     */
    public static <T extends Field<T>> void standardVector(
            Shape shape, Field<T>[] src1, int[] rowPointers, int[] colIndices,
            int size, Field<T>[] src2, int[] indices,
            Field<T>[] dest) {
        // Ensure the matrix and vector have shapes conducive to matrix-vector multiplication.
        int rows1 = shape.get(0);
        ValidateParameters.ensureEquals(shape.get(1), size);
        ValidateParameters.ensureEquals(dest.length, size);
        final Field<T> ZERO = (src1.length > 0) ? src1[0].getZero() : null;
        Arrays.fill(dest, ZERO);

        // Iterate over the non-zero elements of the sparse vector.
        for (int k=0, src2Nnz = src2.length; k < src2Nnz; k++) {
            int col = indices[k];
            Field<T> val = src2[k];

            // Perform multiplication only for the non-zero elements.
            for (int i=0; i<rows1; i++) {
                int start = rowPointers[i];
                int stop = rowPointers[i + 1];
                Field<T> sum = dest[i];

                for (int aIndex=start; aIndex < stop; aIndex++) {
                    int aCol = colIndices[aIndex];

                    if (aCol == col)
                        sum = sum.add(val.mult((T) src1[aIndex]));
                }

                dest[i] = sum;
            }
        }
    }
}
