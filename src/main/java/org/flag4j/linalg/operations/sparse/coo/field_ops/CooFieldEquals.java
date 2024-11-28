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

package org.flag4j.linalg.operations.sparse.coo.field_ops;

import org.flag4j.algebraic_structures.Pair;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.algebraic_structures.rings.Ring;
import org.flag4j.arrays.backend.field.AbstractCooFieldMatrix;
import org.flag4j.arrays.backend.field.AbstractCooFieldTensor;
import org.flag4j.arrays.backend.field.AbstractCooFieldVector;
import org.flag4j.linalg.operations.common.real.RealProperties;
import org.flag4j.linalg.operations.common.ring_ops.RingProperties;
import org.flag4j.util.ErrorMessages;

import java.util.*;

/**
 * <p>This utility class contains methods for checking the equality, or approximately equal, of sparse COO tensors whose data are
 * {@link Field field} elements.
 */
public final class CooFieldEquals {

    private CooFieldEquals() {
        // Hide default constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if two real sparse tensors are real. Assumes the indices of each sparse tensor are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First tensor in the equality check.
     * @param b Second tensor in the equality check.
     * @return True if the tensors are equal. False otherwise.
     */
    public static <T extends Field<T>> boolean cooTensorEquals(
            AbstractCooFieldTensor<?, ?, T> a,
            AbstractCooFieldTensor<?, ?, T> b) {
        // Early returns if possible.
        if(a == b) return true;
        if(a==null || b==null || !a.shape.equals(b.shape)) return false;

        List<Field<T>> aEntries = new ArrayList(a.nnz);
        List<int[]> aIndices = new ArrayList<>(a.nnz);

        List<Field<T>> bEntries = new ArrayList(b.nnz);
        List<int[]> bIndices = new ArrayList(b.nnz);

        for(int i=0; i<a.nnz; i++) {
            if(a.data[i] == null) return false;

            if(!a.data[i].isZero()) {
                aEntries.add((T) a.data[i]);
                aIndices.add(a.indices[i]);
            }
        }

        for(int i=0; i<b.nnz; i++) {
            if(b.data[i] == null) return false;

            if(!b.data[i].isZero()) {
                bEntries.add((T) b.data[i]);
                bIndices.add(b.indices[i]);
            }
        }

        return aEntries.equals(bEntries) && Arrays.deepEquals(aIndices.toArray(new int[0][]), bIndices.toArray(new int[0][]));
    }


    /**
     * Checks if two real sparse matrices are real. Assumes the indices of each sparse matrix are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First matrix in the equality check.
     * @param b Second matrix in the equality check.
     * @return True if the matrices are equal. False otherwise.
     */
    public static <T extends Field<T>> boolean cooMatrixEquals(
            AbstractCooFieldMatrix<?, ?, ?, T> a,
            AbstractCooFieldMatrix<?, ?, ?, T> b) {
        // Early return if possible.
        if (a == b) return true;
        if (a == null || b == null || !a.shape.equals(b.shape)) return false;

        // Use maps to store non-zero values by their row and column indices.
        Map<Pair<Integer>, Field<T>> nonZeroMapA = new HashMap<>();
        for (int i = 0; i < a.nnz; i++) {
            if (a.data[i] == null) return false;
            if (!a.data[i].isZero())
                nonZeroMapA.put(new Pair<>(a.rowIndices[i], a.colIndices[i]), a.data[i]);
        }

        // Compare with matrix b.
        for (int i = 0; i < b.nnz; i++) {
            if (b.data[i] == null) return false;

            Pair<Integer> key = new Pair<>(b.rowIndices[i], b.colIndices[i]);
            Field<T> valueB = b.data[i];

            if (!valueB.isZero()) {
                // If valueB is non-zero, check against matrix a.
                Field<T> valueA = nonZeroMapA.remove(key);
                if (valueA == null || !valueA.equals(valueB)) return false;
            } else {
                // If valueB is zero, ensure first matrix also has zero or does not contain the key.
                if (nonZeroMapA.containsKey(key)) return false;
            }
        }

        // If nonZeroMapA is not empty, then matrix a has non-zero values that matrix b does not.
        return nonZeroMapA.isEmpty();
    }


    /**
     * Checks if two real sparse vectors are real. Assumes the indices of each sparse vector are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static <T extends Field<T>> boolean cooVectorEquals(
            AbstractCooFieldVector<?, ?, ?, ?, T> a,
            AbstractCooFieldVector<?, ?, ?, ?, T> b) {
        // Early returns if possible.
        if(a == b) return true;
        if(a==null || b==null || !a.shape.equals(b.shape)) return false;

        List<Field<T>> aEntries = new ArrayList<>(a.nnz);
        List<Integer> aIndices = new ArrayList<>(a.nnz);

        List<Field<T>> bEntries = new ArrayList<>(b.nnz);
        List<Integer> bIndices = new ArrayList<>(b.nnz);

        for(int i=0; i<a.nnz; i++) {
            if(a.data[i] == null) return false;

            if(!a.data[i].isZero()) {
                aEntries.add(a.data[i]);
                aIndices.add(a.indices[i]);
            }
        }

        for(int i=0; i<b.nnz; i++) {
            if(b.data[i] == null) return false;

            if(!b.data[i].isZero()) {
                bEntries.add(b.data[i]);
                bIndices.add(b.indices[i]);
            }
        }

        return aEntries.equals(bEntries) && aIndices.equals(bIndices);
    }


    /**
     * Checks that all non-zero data are "close" according to {@link RealProperties#allClose(double[], double[])} and
     *      * all indices are the same.
     * @param src1 First matrix in comparison.
     * @param src2 Second matrix in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all data are "close". Otherwise, false.
     */
    public static <T extends Field<T>> boolean allClose(AbstractCooFieldMatrix<?, ?, ?, T> src1,
                                                        AbstractCooFieldMatrix<?, ?, ?, T> src2,
                                                        double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.rowIndices, src2.rowIndices)
                && Arrays.equals(src1.colIndices, src2.colIndices)
                && RingProperties.allClose(src1.data, src2.data, relTol, absTol);
    }


    /**
     * Checks that all non-zero data are "close" according to
     * 
     * {@link RingProperties#allClose(Ring[], Ring[], double, double)} and all indices
     * are the same.
     * @param src1 First tensor in comparison.
     * @param src2 Second tensor in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all data are "close". Otherwise, false.
     */
    public static <T extends Field<T>> boolean allClose(AbstractCooFieldTensor<?, ?, T> src1,
                                                        AbstractCooFieldTensor<?, ?, T> src2,
                                                        double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.deepEquals(src1.indices, src2.indices)
                && RingProperties.allClose(src1.data, src2.data, relTol, absTol);
    }


    /**
     * Checks that all non-zero data are "close" according to
     * {@link RingProperties#allClose(Field[], Field[])} and all indices are the same.
     * @param src1 First vector in comparison.
     * @param src2 Second vector in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all data are "close". Otherwise, false.
     */
    public static <T extends Field<T>> boolean allClose(AbstractCooFieldVector<?, ?, ?, ?, T> src1,
                                                        AbstractCooFieldVector<?, ?, ?, ?, T> src2,
                                                        double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.indices, src2.indices)
                && RingProperties.allClose(src1.data, src2.data, relTol, absTol);
    }
}
