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

package org.flag4j.operations.sparse.csr.real;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;


/**
 * This class contains low-level implementations for element-wise operations on CSR matrices.
 */
public final class RealCsrOperations {

    private RealCsrOperations() {
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
     * @param uOpp Unary operator for use with binary operations which are not commutative such as subtraction. If the operation is
     * commutative this should be {@code null}. If the binary operation is not commutative, it needs to be decomposable to one
     * commutative binary operation {@code opp} and one unary operation {@code uOpp} such that it is equivalent to
     * {@code opp.apply(x, uOpp.apply(y))}.
     * @return The result of applying the specified binary operation to <code>src1</code> and <code>src2</code>
     * element-wise.
     * @throws IllegalArgumentException If <code>src1</code> and <code>src2</code> do not have the same shape.
     */
    public static CsrMatrix applyBinOpp(CsrMatrix src1, CsrMatrix src2,
                                        BinaryOperator<Double> opp,
                                        UnaryOperator<Double> uOpp) {
        ParameterChecks.assertEqualShape(src1.shape, src2.shape);

        List<Double> dest = new ArrayList<>();
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
                    dest.add(src1.entries[rowPtr1]);
                    colIndices.add(col1);
                    rowPtr1++;
                } else {
                    if(uOpp!=null) dest.add(uOpp.apply(src2.entries[rowPtr2]));
                    else dest.add(src2.entries[rowPtr2]);
                    colIndices.add(col2);
                    rowPtr2++;
                }

                rowPointers[i+1]++;
            }

            while(rowPtr1 < src1.rowPointers[i+1]) {
                dest.add(src1.entries[rowPtr1]);
                colIndices.add(src1.colIndices[rowPtr1]);
                rowPtr1++;
                rowPointers[i+1]++;
            }

            while(rowPtr2 < src2.rowPointers[i+1]) {
                if(uOpp!=null) dest.add(uOpp.apply(src2.entries[rowPtr2]));
                else dest.add(src2.entries[rowPtr2]);
                colIndices.add(src2.colIndices[rowPtr2]);
                rowPtr2++;
                rowPointers[i+1]++;
            }
        }

        // Accumulate row pointers.
        for(int i=1; i<rowPointers.length; i++) {
            rowPointers[i] += rowPointers[i-1];
        }

        return new CsrMatrix(src1.shape,
                ArrayUtils.fromDoubleList(dest),
                rowPointers,
                ArrayUtils.fromIntegerList(colIndices)
        );
    }





    /**
     * Transposes a sparse CSR matrix.
     * @param src The matrix to transpose.
     * @return The transpose of the {@code src} matrix.
     */
    public static CsrMatrix transpose(CsrMatrix src) {

        double[] dest = new double[src.entries.length];
        int[] rowPtrs = new int[src.numCols+1];
        int[] colIndices = new int[src.entries.length];

        // Count number of entries in each row.
        for(int i=0; i<src.colIndices.length; i++) {
            rowPtrs[src.colIndices[i] + 1]++;
        }

        // Accumulate the row counts.
        for(int i=1; i<rowPtrs.length; i++) {
            rowPtrs[i] += rowPtrs[i-1];
        }

        int[] tempPos = java.util.Arrays.copyOf(rowPtrs, rowPtrs.length);

        // Fill in the values for the transposed matrix
        for (int row = 0; row < src.numRows; row++) {
            for (int i = src.rowPointers[row]; i < src.rowPointers[row + 1]; i++) {
                int col = src.colIndices[i];
                int pos = tempPos[col];
                dest[pos] = src.entries[i];
                colIndices[pos] = row;
                tempPos[col]++;
            }
        }

        return new CsrMatrix(src.numCols, src.numRows, dest, rowPtrs, colIndices);
    }


    /**
     * Gets a specified slice of a CSR matrix.
     *
     * @param src Sparse CSR matrix to extract slice from.
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd   Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd   Ending row index of slice (exclusive).
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException       If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    public static CsrMatrix getSlice(CsrMatrix src, int rowStart, int rowEnd, int colStart, int colEnd) {
        List<Double> slice = new ArrayList<>();
        List<Integer> sliceRowIndices = new ArrayList<>();
        List<Integer> sliceColIndices = new ArrayList<>();

        // Efficiently construct COO matrix then convert to a CSR matrix.
        int rowStop = rowEnd-1;
        for(int i=rowStart; i<rowStop; i++) {
            // Beginning and ending indices for the row.
            int begin = src.rowPointers[i];
            int end = src.rowPointers[i+1];

            for(int j=begin; j<end; j++) {
                int col = src.colIndices[j];

                // Add value if it is within the slice.
                if(col >= colStart && col < colEnd) {
                    slice.add(src.entries[j]);
                    sliceRowIndices.add(i);
                    sliceColIndices.add(col);
                }
            }
        }

        return new CooMatrix(
                new Shape(rowEnd-rowStart, colEnd-colStart),
                slice, sliceRowIndices, sliceColIndices).toCsr();
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param src1 CSR matrix to add vector to each column of.
     * @param src2 Vector to add to each column of this matrix.
     * @return The result of adding the vector src2 to each column of this matrix.
     */
    public static Matrix addToEachCol(CsrMatrix src1, Vector src2) {
        ParameterChecks.assertEquals(src1.numRows, src2.size);
        Matrix sum = src2.repeat(src1.numCols, 1);

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];
            int rowOffset = i*sum.numCols;

            for(int j=rowStart; j<rowEnd; j++) {
                sum.entries[rowOffset + src1.colIndices[j]] += src1.entries[j];
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param src1 CSR matrix to add vector to each column of.
     * @param src2 Vector to add to each column of this matrix.
     * @return The result of adding the vector src2 to each column of this matrix.
     */
    public static Matrix addToEachCol(CsrMatrix src1, CooVector src2) {
        ParameterChecks.assertEquals(src1.numRows, src2.size);
        Matrix sum = src2.repeat(src1.numCols, 1).toDense();

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];
            int rowOffset = i*sum.numCols;

            for(int j=rowStart; j<rowEnd; j++) {
                sum.entries[rowOffset + src1.colIndices[j]] += src1.entries[j];
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param src1 CSR matrix to add vector to each column of.
     * @param src2 Vector to add to each column of this matrix.
     * @return The result of adding the vector src2 to each column of this matrix.
     */
    public static CMatrix addToEachCol(CsrMatrix src1, CVector src2) {
        ParameterChecks.assertEquals(src1.numRows, src2.size);
        CMatrix sum = src2.repeat(src1.numCols, 1);

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];
            int rowOffset = i*sum.numCols;

            for(int j=rowStart; j<rowEnd; j++) {
                int idx = rowOffset + src1.colIndices[j];
                sum.entries[idx] = sum.entries[idx].add(src1.entries[j]);
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param src1 CSR matrix to add vector to each column of.
     * @param src2 Vector to add to each column of this matrix.
     * @return The result of adding the vector src2 to each column of this matrix.
     */
    public static CMatrix addToEachCol(CsrMatrix src1, CooCVector src2) {
        ParameterChecks.assertEquals(src1.numRows, src2.size);
        CMatrix sum = src2.repeat(src1.numCols, 1).toDense();

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];
            int rowOffset = i*sum.numCols;

            for(int j=rowStart; j<rowEnd; j++) {
                int idx = rowOffset + src1.colIndices[j];
                sum.entries[idx] = sum.entries[idx].add(src1.entries[j]);
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param src1 CSR matrix to add vector to each row of.
     * @param src2 Vector to add to each row of this matrix.
     * @return The result of adding the vector src2 to each row of this matrix.
     */
    public static Matrix addToEachRow(CsrMatrix src1, Vector src2) {
        ParameterChecks.assertEquals(src1.numCols, src2.size);
        Matrix sum = src2.repeat(src1.numRows, 0);

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];

            for(int j=rowStart; j<rowEnd; j++) {
                sum.entries[i*sum.numCols + src1.colIndices[j]] += src1.entries[j];
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param src1 CSR matrix to add vector to each row of.
     * @param src2 Vector to add to each row of this matrix.
     * @return The result of adding the vector src2 to each row of this matrix.
     */
    public static Matrix addToEachRow(CsrMatrix src1, CooVector src2) {
        ParameterChecks.assertEquals(src1.numCols, src2.size);
        Matrix sum = src2.repeat(src1.numRows, 0).toDense();

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];

            for(int j=rowStart; j<rowEnd; j++) {
                sum.entries[i*sum.numCols + src1.colIndices[j]] += src1.entries[j];
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param src1 CSR matrix to add vector to each row of.
     * @param src2 Vector to add to each row of this matrix.
     * @return The result of adding the vector src2 to each row of this matrix.
     */
    public static CMatrix addToEachRow(CsrMatrix src1, CVector src2) {
        ParameterChecks.assertEquals(src1.numCols, src2.size);
        CMatrix sum = src2.repeat(src1.numRows, 0);

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];
            int rowOffset = i*sum.numCols;

            for(int j=rowStart; j<rowEnd; j++) {
                int idx = rowOffset + src1.colIndices[j];
                sum.entries[idx] = sum.entries[idx].add(src1.entries[j]);
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param src1 CSR matrix to add vector to each row of.
     * @param src2 Vector to add to each row of this matrix.
     * @return The result of adding the vector src2 to each row of this matrix.
     */
    public static CMatrix addToEachRow(CsrMatrix src1, CooCVector src2) {
        ParameterChecks.assertEquals(src1.numCols, src2.size);
        CMatrix sum = src2.repeat(src1.numRows, 0).toDense();

        for(int i=0; i<src1.numRows; i++) {
            int rowStart = src1.rowPointers[i];
            int rowEnd = src1.rowPointers[i+1];
            int rowOffset = i*sum.numCols;

            for(int j=rowStart; j<rowEnd; j++) {
                int idx = rowOffset + src1.colIndices[j];
                sum.entries[idx] = sum.entries[idx].add(src1.entries[j]);
            }
        }

        return sum;
    }
}





