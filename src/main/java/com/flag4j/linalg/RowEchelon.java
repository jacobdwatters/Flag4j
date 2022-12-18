/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

/**
 * This class contains several methods for computing row echelon, reduced row echelon, and extended reduced row echelon
 * forms of a matrix.
 */
public class RowEchelon {

//    /**
//     * Computes row echelon form of a Matrix.
//     * @param A The matrix for which to compute the row echelon form.
//     * @return A matrix in row echelon form which is row-equivalent to this matrix.
//     */
//    public static Matrix ref(Matrix A) {
//        // TODO:
//        return null;
//    }
//
//
//    /**
//     * Computes row echelon form of a Matrix.
//     * @param A The matrix for which to compute the row echelon form.
//     * @return A matrix in row echelon form which is row-equivalent to this matrix.
//     */
//    public static SparseMatrix ref(SparseMatrix A) {
//        /*TODO*/
//        return null;
//    }
//
//
//    /**
//     * Computes row echelon form of a Matrix.
//     * @param A The matrix for which to compute the row echelon form.
//     * @return A matrix in row echelon form which is row-equivalent to this matrix.
//     */
//    public static CMatrix ref(CMatrix A) {
//        // TODO:
//        return null;
//    }
//
//
//    /**
//     * Computes row echelon form of a Matrix.
//     * @param A The matrix for which to compute the row echelon form.
//     * @return A matrix in row echelon form which is row-equivalent to this matrix.
//     */
//    public static SparseCMatrix ref(SparseCMatrix A) {
//        // TODO:
//        return null;
//    }
//
//
//    /**
//     * Computes the reduced row echelon form of a matrix. By default, this method uses partial pivoting.
//     * To use full or no pivoting see {@link #rref(Matrix, int)}.
//     * @param A The matrix for which to compute the reduced row echelon form.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
//     */
//    public static Matrix rref(Matrix A) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the reduced row echelon form of a matrix. By default, this method uses partial pivoting.
//     * To use full or no pivoting see {@link #rref(Matrix, int)}.
//     * @param A The matrix for which to compute the reduced row echelon form.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
//     */
//    public static SparseMatrix rref(SparseMatrix A) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the reduced row echelon form of a matrix. By default, this method uses partial pivoting.
//     * To use full or no pivoting see {@link #rref(CMatrix, int)}.
//     * @param A The matrix for which to compute the reduced row echelon form.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
//     */
//    public static CMatrix rref(CMatrix A) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the reduced row echelon form of a matrix. By default, this method uses partial pivoting.
//     * To use full or no pivoting see {@link #rref(SparseCMatrix, int)}.
//     * @param A The matrix for which to compute the reduced row echelon form.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
//     */
//    public static SparseCMatrix rref(SparseCMatrix A) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the reduced row echelon form of a matrix possibly with partial or full pivoting.
//     * Also see {@link #rref(Matrix, int)} for partial pivoting.
//     * @param A The matrix for which to compute the reduced row echelon form.
//     * @param pivoting Indicates what pivoting method to use. <br>
//     *        - If pivoting=0, no pivoting is used.<br>
//     *        - If pivoting=1, partial pivoting is used.<br>
//     *        - If pivoting=2, full pivoting is used.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
//     * @throws IllegalArgumentException if pivoting is not 0, 1, or 2.
//     */
//    public static Matrix rref(Matrix A, int pivoting) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the reduced row echelon form of a matrix possibly with partial or full pivoting.
//     * Also see {@link #rref(SparseMatrix, int)} for partial pivoting.
//     * @param A The matrix for which to compute the reduced row echelon form.
//     * @param pivoting Indicates what pivoting method to use. <br>
//     *        - If pivoting=0, no pivoting is used.<br>
//     *        - If pivoting=1, partial pivoting is used.<br>
//     *        - If pivoting=2, full pivoting is used.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
//     * @throws IllegalArgumentException if pivoting is not 0, 1, or 2.
//     */
//    public static SparseMatrix rref(SparseMatrix A, int pivoting) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the reduced row echelon form of a matrix possibly with partial or full pivoting.
//     * Also see {@link #rref(CMatrix, int)} for partial pivoting.
//     * @param A The matrix for which to compute the reduced row echelon form.
//     * @param pivoting Indicates what pivoting method to use. <br>
//     *        - If pivoting=0, no pivoting is used.<br>
//     *        - If pivoting=1, partial pivoting is used.<br>
//     *        - If pivoting=2, full pivoting is used.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
//     * @throws IllegalArgumentException if pivoting is not 0, 1, or 2.
//     */
//    public static CMatrix rref(CMatrix A, int pivoting) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the reduced row echelon form of a matrix possibly with partial or full pivoting.
//     * Also see {@link #rref(SparseCMatrix, int)} for partial pivoting.
//     * @param A The matrix for which to compute the reduced row echelon form.
//     * @param pivoting Indicates what pivoting method to use. <br>
//     *        - If pivoting=0, no pivoting is used.<br>
//     *        - If pivoting=1, partial pivoting is used.<br>
//     *        - If pivoting=2, full pivoting is used.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix.
//     * @throws IllegalArgumentException if pivoting is not 0, 1, or 2.
//     */
//    public static SparseCMatrix rref(SparseCMatrix A, int pivoting) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the extended reduced row echelon form of a matrix. This is equivalent to <code>{@link #rref(Matrix) rref(A.augment(Matrix.I(A.numRows())))}</code>
//     * @param A Matrix for which to compute extended reduced row echelon form of.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix augmented with the
//     * appropriately sized identity matrix.
//     */
//    public static Matrix erref(Matrix A) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the extended reduced row echelon form of a matrix. This is equivalent to <code>{@link #rref(SparseMatrix) rref(A.augment(Matrix.I(A.numRows())))}</code>
//     * @param A Matrix for which to compute extended reduced row echelon form of.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix augmented with the
//     * appropriately sized identity matrix.
//     */
//    public static SparseMatrix erref(SparseMatrix A) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the extended reduced row echelon form of a matrix. This is equivalent to <code>{@link #rref(CMatrix) rref(A.augment(Matrix.I(A.numRows())))}</code>
//     * @param A Matrix for which to compute extended reduced row echelon form of.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix augmented with the
//     * appropriately sized identity matrix.
//     */
//    public static CMatrix erref(CMatrix A) {/*TODO*/return null;}
//
//
//    /**
//     * Computes the extended reduced row echelon form of a matrix. This is equivalent to <code>{@link #rref(SparseCMatrix) rref(A.augment(Matrix.I(A.numRows())))}</code>
//     * @param A Matrix for which to compute extended reduced row echelon form of.
//     * @return A matrix in reduced row echelon form which is row-equivalent to this matrix augmented with the
//     * appropriately sized identity matrix.
//     */
//    public static SparseCMatrix erref(SparseCMatrix A) {/*TODO*/return null;}
}
