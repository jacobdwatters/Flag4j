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

package org.flag4j.operations.sparse.coo.field_ops;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.sparse.CooFieldMatrix;
import org.flag4j.arrays.sparse.CooFieldTensor;
import org.flag4j.arrays.sparse.CooFieldVector;
import org.flag4j.operations.common.field_ops.CompareField;
import org.flag4j.operations.common.real.RealProperties;
import org.flag4j.util.ErrorMessages;

import java.util.*;

/**
 * This utility class contains methods for checking the equality of sparse tensors whose entries are {@link Field field} elements.
 */
public final class SparseFieldEquals {

    private SparseFieldEquals() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if two real sparse tensors are real. Assumes the indices of each sparse tensor are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First tensor in the equality check.
     * @param b Second tensor in the equality check.
     * @return True if the tensors are equal. False otherwise.
     */
    public static <T extends Field<T>> boolean cooTensorEquals(CooFieldTensor<T> a, CooFieldTensor<T> b) {
        // Early returns if possible.
        if(a == b) return true;
        if(a==null || b==null || !a.shape.equals(b.shape)) return false;

        List<T> aEntries = new ArrayList(a.nnz);
        List<int[]> aIndices = new ArrayList<>(a.nnz);

        List<T> bEntries = new ArrayList(b.nnz);
        List<int[]> bIndices = new ArrayList(b.nnz);

        for(int i=0; i<a.nnz; i++) {
            if(a.entries[i] == null) return false;

            if(!a.entries[i].isZero()) {
                aEntries.add(a.entries[i]);
                aIndices.add(a.indices[i]);
            }
        }

        for(int i=0; i<b.nnz; i++) {
            if(b.entries[i] == null) return false;

            if(!b.entries[i].isZero()) {
                bEntries.add(b.entries[i]);
                bIndices.add(b.indices[i]);
            }
        }

        return aEntries.equals(bEntries) && Arrays.deepEquals(aIndices.toArray(new int[0][]), bIndices.toArray(new int[0][]));
    }


    /**
     * Checks if two real sparse matrices are real. Assumes the indices of each sparse matrix are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param src1 First matrix in the equality check.
     * @param src2 Second matrix in the equality check.
     * @return True if the matrices are equal. False otherwise.
     */
    public static <T extends Field<T>> boolean cooMatrixEquals(CooFieldMatrix<T> src1, CooFieldMatrix<T> src2) {
        // Check if shapes are equal.
        if (!src1.shape.equals(src2.shape)) return false;

        // Counters for explicitly stored zero values.
        int aZeroCount = 0;
        int bZeroCount = 0;

        // Create src1 HashMap ignoring the explicit zeros in first matrix.
        Map<Integer, T> nonZeroMapA = new HashMap<>();
        for (int i=0; i < src1.nnz; i++) {
            if(src1.entries[i] == null) return false;

            if (!src1.entries[i].isZero()) {
                nonZeroMapA.put(i-aZeroCount, src1.entries[i]);
            } else{
                aZeroCount++;
            }
        }

        // Iterate over matrix src2's entries and compare with matrix src1's entries.
        for (int i=0; i<src2.nnz; i++) {
            if(src2.entries[i] == null) return false;

            int key = i-bZeroCount;
            T valueB = src2.entries[i];

            // If valueB is non-zero, check against matrix src1.
            if(!valueB.isZero()) {
                T valueA = nonZeroMapA.remove(key);

                // If valueA is null or values differ, matrices are not equal.
                if (valueA == null || !valueA.equals(valueB)) return false;
            } else {
                bZeroCount++;

                // If valueB is zero, ensure matrix first matrix also has zero or does not contain the key.
                if (nonZeroMapA.containsKey(key)) {
                    return false;
                }
            }
        }

        if(nonZeroMapA.size() != 0) return false; // Then matrix src1 must have contained src1 non-zero value that src2 did not.

        // All checks passed, matrices are equal.
        return true;
    }


    /**
     * Checks if two real sparse vectors are real. Assumes the indices of each sparse vector are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static <T extends Field<T>> boolean cooVectorEquals(CooFieldVector<T> a, CooFieldVector<T> b) {
        // Early returns if possible.
        if(a == b) return true;
        if(a==null || b==null || !a.shape.equals(b.shape)) return false;

        List<T> aEntries = new ArrayList<>(a.nnz);
        List<Integer> aIndices = new ArrayList<>(a.nnz);

        List<T> bEntries = new ArrayList<>(b.nnz);
        List<Integer> bIndices = new ArrayList<>(b.nnz);

        for(int i=0; i<a.nnz; i++) {
            if(a.entries[i] == null) return false;

            if(!a.entries[i].isZero()) {
                aEntries.add(a.entries[i]);
                aIndices.add(a.indices[i]);
            }
        }

        for(int i=0; i<b.nnz; i++) {
            if(b.entries[i] == null) return false;

            if(!b.entries[i].isZero()) {
                bEntries.add(b.entries[i]);
                bIndices.add(b.indices[i]);
            }
        }

        return aEntries.equals(bEntries) && aIndices.equals(bIndices);
    }


    /**
     * Checks that all non-zero entries are "close" according to {@link RealProperties#allClose(double[], double[])} and
     *      * all indices are the same.
     * @param src1 First matrix in comparison.
     * @param src2 Second matrix in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all entries are "close". Otherwise, false.
     */
    public static <T extends Field<T>> boolean allCloseMatrix(CooFieldMatrix<T> src1, CooFieldMatrix<T> src2,
                                                              double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.rowIndices, src2.rowIndices)
                && Arrays.equals(src1.colIndices, src2.colIndices)
                && CompareField.allClose(src1.entries, src2.entries, relTol, absTol);
    }


    /**
     * Checks that all non-zero entries are "close" according to {@link RealProperties#allClose(double[], double[], double, double)} and
     * all indices are the same.
     * @param src1 First tensor in comparison.
     * @param src2 Second tensor in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all entries are "close". Otherwise, false.
     */
    public static <T extends Field<T>> boolean allCloseTensor(CooFieldTensor<T> src1, CooFieldTensor<T> src2,
                                                              double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.deepEquals(src1.indices, src2.indices)
                && CompareField.allClose(src1.entries, src2.entries, relTol, absTol);
    }


    /**
     * Checks that all non-zero entries are "close" according to {@link RealProperties#allClose(double[], double[])} and
     * all indices are the same.
     * @param src1 First vector in comparison.
     * @param src2 Second vector in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all entries are "close". Otherwise, false.
     */
    public static <T extends Field<T>> boolean allCloseVector(CooFieldVector<T> src1, CooFieldVector<T> src2,
                                                              double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.indices, src2.indices)
                && CompareField.allClose(src1.entries, src2.entries, relTol, absTol);
    }
}