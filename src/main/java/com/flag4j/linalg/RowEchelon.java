/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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

package com.flag4j.linalg;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.decompositions.ComplexLUDecomposition;
import com.flag4j.linalg.decompositions.RealLUDecomposition;

/**
 * This class contains several methods for computing row echelon, reduced row echelon, and extended reduced row echelon
 * forms of a matrix.
 */
public class RowEchelon {

    /**
     * Computes a row echelon form of a Matrix. For a reduced row echelon form use {@link #rref(Matrix)}.
     * @param A The matrix for which to compute the row echelon form.
     * @return A matrix in row echelon form which is row-equivalent to the matrix {@code A}.
     */
    public static Matrix ref(Matrix A) {
        return new RealLUDecomposition().decompose(A).getU();
    }


    /**
     * Computes a row echelon form of a Matrix. For a reduced row echelon form use {@link #rref(CMatrix)}.
     * @param A The matrix for which to compute the row echelon form.
     * @return A matrix in row echelon form which is row-equivalent to the matrix {@code A}.
     */
    public static CMatrix ref(CMatrix A) {
        return new ComplexLUDecomposition().decompose(A).getU();
    }


    /**
     * Computes the reduced row echelon form of a matrix. For a non-reduced row echelon form see {@link #ref(Matrix)}.
     * @param A The matrix for which to compute the reduced row echelon form.
     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
     */
    public static Matrix rref(Matrix A) {
        Matrix U = new RealLUDecomposition().decompose(A).getU();
        int colStop = Math.min(U.numCols, U.numRows);
        int pivotRow;
        int iRow;
        double m;
        double pivotValue;

        for(int j=0; j<colStop; j++) {
            pivotRow = j*U.numCols;
            pivotValue = U.entries[pivotRow + j];

            // Scale row so pivot is 1.
            for(int k=j; k<U.numCols; k++) {
                U.entries[pivotRow + k] = U.entries[pivotRow + k]/pivotValue;
            }

            // Zero out column above pivot.
            for(int i=0; i<j; i++) {
                iRow = i*U.numCols;
                m = U.entries[iRow + j];

                for(int k=j; k<U.numCols; k++) {
                    U.entries[iRow + k] -= m*U.entries[pivotRow + k];
                }
            }
        }

        return U;
    }


    /**
     * Computes the reduced row echelon form of a matrix. For a non-reduced row echelon form see {@link #ref(CMatrix)}.
     * @param A The matrix for which to compute the reduced row echelon form.
     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
     */
    public static CMatrix rref(CMatrix A) {
        CMatrix U = new ComplexLUDecomposition().decompose(A).getU();
        int colStop = Math.min(U.numCols, U.numRows);
        int pivotRow;
        int iRow;
        CNumber m;

        for(int j=0; j<colStop; j++) {
            pivotRow = j*U.numCols;
            m = U.entries[pivotRow + j];

            // Scale row so pivot is 1.
            for(int k=j; k<U.numCols; k++) {
                U.entries[pivotRow + k] = U.entries[pivotRow + k].div(m);
            }

            // Zero out column above pivot.
            for(int i=0; i<j; i++) {
                iRow = i*U.numCols;
                m = U.entries[iRow + j];

                for(int k=j; k<U.numCols; k++) {
                    U.entries[iRow + k].subEq(m.mult(U.entries[pivotRow + k]));
                }
            }
        }

        return U;
    }


    /**
     * Computes the extended reduced row echelon form of a matrix. This is equivalent to <code>{@link #rref(Matrix) rref(A.augment(Matrix.I(A.numRows())))}</code>
     * @param A Matrix for which to compute extended reduced row echelon form of.
     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix augmented with the
     * appropriately sized identity matrix.
     */
    public static Matrix erref(Matrix A) {
        return rref(A.augment(Matrix.I(A.numRows)));
    }


    /**
     * Computes the extended reduced row echelon form of a matrix. This is equivalent to <code>{@link #rref(CMatrix) rref(A.augment(Matrix.I(A.numRows())))}</code>
     * @param A Matrix for which to compute extended reduced row echelon form of.
     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix augmented with the
     * appropriately sized identity matrix.
     */
    public static CMatrix erref(CMatrix A) {
        return rref(A.augment(Matrix.I(A.numRows)));
    }
}
