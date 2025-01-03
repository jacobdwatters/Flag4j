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

package org.flag4j.arrays.sparse;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.io.Serializable;
import java.util.Arrays;

/**
 * A permutation matrix is a square matrix containing only zeros and ones such that each row and column have exactly a single
 * one. The identity matrix is a special case of a permutation matrix. Permutation matrices are commonly used to
 * track or apply row/column swaps in a matrix.<br><br>
 *
 * All permutation matrices are {@link Matrix#isOrthogonal() orthogonal}/{@link CMatrix#isUnitary() unitary} meaning
 * their inverse is equal to their transpose.<br><br>
 *
 * When a permutation matrix is left multiplied to a second matrix, it has the result of swapping rows
 * in the second matrix.<br><br>
 *
 * Similarly, when a permutation matrix is right multiplied to another matrix, it has the result of swapping columns in
 * the other matrix.
 */
public class PermutationMatrix implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Tracks row/column swaps within the permutation matrix. For an {@code n-by-n} permutation matrix, this array will
     * have size {@code n}. Each entry of the array represents a 1 in the permutation matrix. The index of an entry
     * corresponds to the row index of the 1, and the value of this array corresponds to the column index of the 1.
     */
    public final int[] swapPointers;
    /**
     * Size of this permutation matrix.
     */
    public final int size;


    /**
     * Creates a permutation matrix which is equivalent to the identity matrix of the specified size.
     * @param size Size of the permutation matrix. That is, the number of rows and columns
     */
    public PermutationMatrix(int size) {
        this.size = size;
        swapPointers = ArrayUtils.intRange(0, size);
    }


    /**
     * Creates a permutation matrix which is equivalent to the identity matrix of the specified size.
     * @param shape Shape of the permutation matrix. That is, the number of rows and columns. Must be a square shape.
     * @throws LinearAlgebraException If {@code shape} is not square.
     */
    public PermutationMatrix(Shape shape) {
        ValidateParameters.ensureSquareMatrix(shape);
        this.size = shape.get(0);
        swapPointers = ArrayUtils.intRange(0, size);
    }


    /**
     * Copy constructor which creates a copy of the {@code src} permutation matrix.
     * @param src The permutation matrix to copy.
     */
    public PermutationMatrix(PermutationMatrix src) {
        this.size = src.size;
        this.swapPointers = src.swapPointers.clone();
    }


    /**
     * Creates a permutation matrix where the position of its ones are specified by a {@link #swapPointers swap pointer}
     * array.
     * @param swapPointers An array which defines row/column swaps within the permutation matrix.
     *                     For an {@code n-by-n} permutation matrix, this array will have size {@code n}.
     *                     Each entry of the array represents a 1 in the permutation matrix. The index of an entry
     *                     corresponds to the row index of the 1, and the value of this array corresponds to
     *                     the column index of the 1. This must be a permutation matrix. However, the validity of this
     *                     is not enforced by this constructor.
     */
    public PermutationMatrix(int[] swapPointers) {
        this.size = swapPointers.length;
        this.swapPointers = swapPointers.clone();
    }


    /**
     * Creates a permutation matrix with the specified column swaps.
     * @param colSwaps Array specifying column swaps. The entry {@code x} at index {@code i} indicates that column
     * {@code i} has been swapped with column {@code x}. Must be a
     * {@link ValidateParameters#ensurePermutation(int...) permutation array}.
     * @return A permutation matrix with the specified column swaps.
     * @throws IllegalArgumentException If {@code colSwaps} is not a
     * {@link ValidateParameters#ensurePermutation(int...) permutation array}.
     */
    public static PermutationMatrix fromColSwaps(int[] colSwaps) {
        int[] rowPerm = new int[colSwaps.length];

        for (int i=0; i<colSwaps.length; i++) {
            rowPerm[colSwaps[i]] = i;
        }

        return new PermutationMatrix(rowPerm);
    }


    /**
     * Creates a copy of this permutation matrix.
     * @return A copy of this permutation matrix.
     */
    public PermutationMatrix copy() {
        return new PermutationMatrix(this);
    }


    /**
     * Checks if this permutation matrix is equal to the given object. A permutation matrix is considered equal to an
     * object if that object is also a permutation matrix and represents the same matrix.
     * @param object Object to compare to this permutation matrix.
     * @return True if {@code b} is a permutation matrix and equivalent to this matrix (in terms of matrix equality).
     */
    @Override
    public boolean equals(Object object) {
        // Check for quick returns.
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        PermutationMatrix src2 = (PermutationMatrix) object;

        return Arrays.equals(swapPointers, src2.swapPointers);
    }


    /**
     * Returns a hashcode for this permutation matrix by calling {@link Arrays#hashCode(int[]) Arrays.hashCode(swapPointers)}.
     * @return The hashcode for this permutation matrix.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(swapPointers);
    }


    /**
     * Left multiplies this permutation matrix to the specified matrix. This will have the effect of swapping rows in
     * the src matrix.
     * @param src The matrix to left multiply this permutation matrix to.
     * @return The result of left multiplying this permutation matrix to the {@code src} matrix.
     * @see #rightMult(Matrix)
     * @throws IllegalArgumentException If the number of rows in {@code src} does not equal the size of this permutation
     * matrix.
     */
    public Matrix leftMult(Matrix src) {
        ValidateParameters.ensureEquals(size, src.numRows);
        double[] destEntries = new double[src.data.length];
        int colIdx;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            colIdx = swapPointers[rowIdx];
            System.arraycopy(src.data, colIdx*src.numCols, destEntries, rowIdx*src.numCols, src.numCols);
        }

        return new Matrix(src.shape, destEntries);
    }


    /**
     * Left multiplies this permutation matrix to the specified vector. This will have the effect of swapping rows in
     * the src vector. The vector will be treated as a column vector.
     * @param src The vector to left multiply this permutation matrix to.
     * @return The result of left multiplying this permutation matrix to the {@code src} vector.
     * @see #rightMult(Vector)
     * @throws IllegalArgumentException If size of {@code src} does not equal the size of this permutation
     * matrix.
     */
    public Vector leftMult(Vector src) {
        ValidateParameters.ensureEquals(size, src.size);
        double[] destEntries = new double[src.data.length];

        for(int rowIdx=0; rowIdx<size; rowIdx++)
            destEntries[rowIdx] = src.data[swapPointers[rowIdx]];

        return new Vector(destEntries);
    }


    /**
     * Left multiplies this permutation matrix to the specified matrix. This will have the effect of swapping rows in
     * the src matrix.
     * @param src The matrix to left multiply this permutation matrix to.
     * @return The result of left multiplying this permutation matrix to the {@code src} matrix.
     * @see #rightMult(CMatrix)
     * @throws IllegalArgumentException If the number of rows in {@code src} does not equal the size of this permutation
     * matrix.
     */
    public CMatrix leftMult(CMatrix src) {
        ValidateParameters.ensureEquals(size, src.numRows);
        Complex128[] destEntries = new Complex128[src.data.length];
        int colIdx;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            colIdx = swapPointers[rowIdx];
            System.arraycopy(src.data, colIdx*src.numCols, destEntries, rowIdx*src.numCols, src.numCols);
        }

        return new CMatrix(src.shape, destEntries);
    }


    /**
     * Left multiplies this permutation matrix to the specified vector. This will have the effect of swapping rows in
     * the src vector. The vector will be treated as a column vector.
     * @param src The vector to left multiply this permutation matrix to.
     * @return The result of left multiplying this permutation matrix to the {@code src} vector.
     * @see #rightMult(CVector)
     * @throws IllegalArgumentException If size of {@code src} does not equal the size of this permutation
     * matrix.
     */
    public CVector leftMult(CVector src) {
        ValidateParameters.ensureEquals(size, src.size);
        Complex128[] destEntries = new Complex128[src.data.length];

        for(int rowIdx=0; rowIdx<size; rowIdx++)
            destEntries[rowIdx] = src.data[swapPointers[rowIdx]];

        return new CVector(destEntries);
    }


    /**
     * Right multiplies this permutation matrix to the specified matrix. This is equivalent to swapping columns in the
     * {@code src} matrix.
     * @param src The matrix to right multiply this permutation matrix to.
     * @return The result of right multiplying this permutation matrix to the {@code src} matrix.
     * @throws IllegalArgumentException If the number of columns in {@code src} does not match the size of this
     * permutation matrix.
     * @see #leftMult(Matrix)
     */
    public Matrix rightMult(Matrix src) {
        ValidateParameters.ensureEquals(size, src.numCols);
        double[] destEntries = new double[src.data.length];

        int colIdx;
        int rowOffset;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            colIdx = swapPointers[rowIdx];

            for(int j=0; j<src.numRows; j++) {
                rowOffset = j*src.numCols;
                destEntries[rowOffset + colIdx] = src.data[rowOffset + rowIdx];
            }
        }

        return new Matrix(src.shape, destEntries);
    }


    /**
     * Right multiplies this permutation matrix to the specified vector. This will have the effect of swapping columns in
     * the src vector. The vector will be treated as a row vector.
     * @param src The vector to right multiply this permutation matrix to.
     * @return The result of right multiplying this permutation matrix to the {@code src} vector.
     * @see #leftMult(Vector)
     * @throws IllegalArgumentException If size of {@code src} does not equal the size of this permutation
     * matrix.
     */
    public Vector rightMult(Vector src) {
        // For vectors, left/right multiplication is equivalent since vectors do not have orientation
        // (i.e. row/column vectors.)
        return leftMult(src);
    }


    /**
     * Right multiplies this permutation matrix to the specified matrix. This is equivalent to swapping columns in the
     * {@code src} matrix.
     * @param src The matrix to right multiply this permutation matrix to.
     * @return The result of right multiplying this permutation matrix to the {@code src} matrix.
     * @throws IllegalArgumentException If the number of columns in {@code src} does not match the size of this
     * permutation matrix.
     * @see #leftMult(Matrix)
     */
    public CMatrix rightMult(CMatrix src) {
        ValidateParameters.ensureEquals(size, src.numCols);
        Complex128[] destEntries = new Complex128[src.data.length];
        final int rows = src.numRows;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            int colIdx = swapPointers[rowIdx];

            for(int j=0; j<src.numRows; j++) {
                int rowOffset = j*src.numCols;
                destEntries[rowOffset + colIdx] = src.data[rowOffset + rowIdx];
            }
        }

        return new CMatrix(src.shape, destEntries);
    }


    /**
     * Right multiplies this permutation matrix to the specified vector. This will have the effect of swapping columns in
     * the src vector. The vector will be treated as a row vector.
     * @param src The vector to right multiply this permutation matrix to.
     * @return The result of right multiplying this permutation matrix to the {@code src} vector.
     * @see #leftMult(Vector)
     * @throws IllegalArgumentException If size of {@code src} does not equal the size of this permutation
     * matrix.
     */
    public CVector rightMult(CVector src) {
        // For vectors, left/right multiplication is equivalent since vectors do not have orientation
        // (i.e. row/column vectors.)
        return leftMult(src);
    }


    /**
     * Swaps two rows in this permutation matrix.
     * @param row1 First row to swap in the permutation matrix.
     * @param row2 Second row to swap in the permutation matrix.
     * @throws ArrayIndexOutOfBoundsException If either {@code row1} or {@code row2} is out of bounds of this permutation
     * matrix.
     */
    public void swapRows(int row1, int row2) {
        ArrayUtils.swap(swapPointers, row1, row2);
    }


    /**
     * Swaps two columns in this permutation matrix.
     * @param col1 First column to swap in the permutation matrix.
     * @param col2 Second column to swap in the permutation matrix.
     * @throws ArrayIndexOutOfBoundsException If either {@code col1} or {@code col2} is out of bounds of this permutation
     * matrix.
     */
    public void swapCols(int col1, int col2) {
        ValidateParameters.ensureValidArrayIndices(size, col1, col2);
        // Find locations of data with the given columns.
        int idx1 = ArrayUtils.indexOf(swapPointers, col1);
        int idx2 = ArrayUtils.indexOf(swapPointers, col2);
        ArrayUtils.swap(swapPointers, idx1, idx2); // Swap values.
    }


    /**
     * Permutes rows of this permutation matrix.
     * @param swaps Defines row swaps of this permutation matrix. The entry {@code x} at index {@code i}
     *              represents row {@code i} has been swapped with row {@code x}. This must be a
     *              {@link ValidateParameters#ensurePermutation(int...)  permutation} array.
     * @throws IllegalArgumentException If {@code swaps} is not the same length as the number of rows/columns in this
     * permutation matrix. Or, if {@code swaps} is not a
     * {@link ValidateParameters#ensurePermutation(int...)  permutation} array.
     */
    public void permuteRows(int[] swaps) {
        ValidateParameters.ensurePermutation(swaps);
        ValidateParameters.ensureArrayLengthsEq(swaps.length, swapPointers.length);
        System.arraycopy(swaps, 0, swapPointers, 0, swaps.length);
    }


    /**
     * Computes the inverse/transpose of this permutation matrix.
     * @return The inverse/transpose of this permutation matrix.
     */
    public PermutationMatrix inv() {
        return T();
    }


    /**
     * Computes the transpose/inverse of this permutation matrix.
     * @return The transpose/inverse of this permutation matrix.
     */
    public PermutationMatrix T() {
        int[] transpose = new int[size];

        for(int i=0; i<size; i++)
            transpose[swapPointers[i]] = i;

        return new PermutationMatrix(transpose);
    }


    /**
     * Computes the trace of this permutation matrix.
     * @return The trace of this permutation matrix.
     */
    public int trace() {
        return tr();
    }


    /**
     * Computes the trace of this permutation matrix. Alias for {@link #trace()}.
     * @return The trace of this permutation matrix.
     */
    public int tr() {
        int trace = 0;

        for(int i=0; i<size; i++)
            if(swapPointers[i]==i) trace += 1;

        return trace;
    }


    /**
     * Computes the number of row/column swaps required for this permutation matrix to be converted to the identity matrix.
     * @return The total number of row/column swaps required to convert this permutation matrix to the identity matrix.
     */
    public int computeSwaps() {
        boolean[] visited = new boolean[swapPointers.length];
        int totalSwaps = 0;

        for (int i = 0; i < swapPointers.length; i++) {
            if (!visited[i]) {
                visited[i] = true;

                if (swapPointers[i] != i) {
                    int cycleSize = 1;
                    int next = swapPointers[i];

                    while (next != i) {
                        visited[next] = true;
                        next = swapPointers[next];
                        cycleSize++;
                    }

                    totalSwaps += cycleSize - 1;
                }
            }
        }

        return totalSwaps;
    }


    /**
     * Computes the determinant of this permutation matrix (will be +/- 1).
     * @return The determinant of this permutation matrix (will be +/- 1).
     */
    public int det() {
        // Determinant is 1 for an even number of swaps, and -1 for an odd number of swaps.
        return (computeSwaps() & 1) == 0 ? 1 : -1;
    }


    /**
     * Converts this permutation matrix to a {@link Matrix real dense matrix}.
     * @return A real dense matrix which is equivalent to this permutation matrix.
     */
    public Matrix toDense() {
        double[] entries = new double[size*size];
        int rowOffset = 0;
        int colIdx;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            colIdx = swapPointers[rowIdx];
            entries[rowOffset + colIdx] = 1;
            rowOffset += size;
        }

        return new Matrix(size, size, entries);
    }


    /**
     * Converts this permutation matrix to a human-readable string.
     * @return This permutation matrix represented as a human-readable string.
     */
    public String toString() {
        return "Full Shape=" + new Shape(size, size) + "\n" +
                "swap pointers: " + Arrays.toString(swapPointers);
    }
}
