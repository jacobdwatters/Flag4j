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

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.CooFieldMatrixBase;
import org.flag4j.arrays.backend.CooFieldVectorBase;
import org.flag4j.arrays.backend.DenseFieldMatrixBase;
import org.flag4j.arrays.backend.DenseFieldVectorBase;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * This utility class contains methods for computing operations between two sparse coo
 * {@link Field} vectors.
 */
public final class CooFieldVectorOperations {

    private CooFieldVectorOperations() {
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static <T extends Field<T>> DenseFieldVectorBase<?, ?, ?, T> add(CooFieldVectorBase<?, ?, ?, ?, T> src, T a) {
        Field<T>[] dest = new Field[src.size];
        Arrays.fill(dest, a);

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.indices[i];
            dest[src.indices[i]] = dest[src.indices[i]].add((T) src.entries[i]);
        }

        return src.makeLikeDenseTensor((T[]) dest);
    }


    /**
     * Subtracts a real number from each entry of a sparse vector, including the zero entries.
     * @param src Sparse vector to subtract value from.
     * @param a Value to subtract from the {@code src} sparse vector.
     * @return The result of subtracting the specified value from the sparse vector.
     */
    public static <T extends Field<T>> DenseFieldVectorBase<?, ?, ?, T> sub(CooFieldVectorBase<?, ?, ?, ?, T> src, T a) {
        Field<T>[] dest = new Field[src.size];
        Arrays.fill(dest, a.addInv());

        for(int i=0; i<src.entries.length; i++) {
            int idx = src.indices[i];
            dest[idx] = dest[idx].add((T) src.entries[i]);
        }

        return src.makeLikeDenseTensor((T[]) dest);
    }


    /**
     * Computes the element-wise vector addition between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @return The result of the vector addition.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends CooFieldVectorBase<T, ?, ?, ?, U>, U extends Field<U>> T add(T src1, T src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        List<Field<U>> values = new ArrayList<>(src1.entries.length);
        List<Integer> indices = new ArrayList<>(src1.entries.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(src1.entries[src1Counter].add((U) src2.entries[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(src1.entries[src1Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2.entries[src2Counter]);
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.entries.length) {
            for(int i=src1Counter; i<src1.entries.length; i++) {
                values.add(src1.entries[i]);
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.entries.length) {
            for(int i=src2Counter; i<src2.entries.length; i++) {
                values.add(src2.entries[i]);
                indices.add(src2.indices[i]);
            }
        }

        return src1.makeLikeTensor(src1.size, values, indices);
    }


    /**
     * Computes the element-wise vector subtraction between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends Field<T>> CooFieldVectorBase<?, ?, ?, ?, T> sub(CooFieldVectorBase<?, ?, ?, ?, T> src1,
                                                                       CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        List<Field<T>> values = new ArrayList<>(src1.entries.length);
        List<Integer> indices = new ArrayList<>(src1.entries.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(src1.entries[src1Counter].sub((T) src2.entries[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(src1.entries[src1Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2.entries[src2Counter].addInv());
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.entries.length) {
            for(int i=src1Counter; i<src1.entries.length; i++) {
                values.add(src1.entries[i]);
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.entries.length) {
            for(int i=src2Counter; i<src2.entries.length; i++) {
                values.add(src2.entries[i].addInv());
                indices.add(src2.indices[i]);
            }
        }

        return src2.makeLikeTensor(src1.size, values, indices);
    }


    /**
     * Computes the element-wise vector multiplication between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the multiplication. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the multiplication. Indices assumed to be sorted lexicographically.
     * @return The result of the vector multiplication.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends Field<T>> CooFieldVectorBase<?, ?, ?, ?, T> elemMult(CooFieldVectorBase<?, ?, ?, ?, T> src1,
                                                                            CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        List<Field<T>> values = new ArrayList<>(src1.entries.length);
        List<Integer> indices = new ArrayList<>(src1.entries.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                values.add(src1.entries[src1Counter].mult((T) src2.entries[src2Counter]));
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return src2.makeLikeTensor(src1.size, values, indices);
    }


    /**
     * Computes the inner product of two complex sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @return The result of the vector inner product.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends Field<T>> T inner(CooFieldVectorBase<?, ?, ?, ?, T> src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        T product = null;
        if(src1.nnz > 0) product = src1.entries[0].getZero();
        else if(src2.nnz > 0) product = src2.entries[0].getZero();

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                product = product.add(src1.entries[src1Counter].mult(src2.entries[src2Counter].conj()));
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return product;
    }


    /**
     * Computes the dot product of two complex sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the dot product. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the dot product. Indices assumed to be sorted lexicographically.
     * @return The result of the vector dot product.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends Field<T>> T dot(CooFieldVectorBase<?, ?, ?, ?, T> src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        T product = null;
        if(src1.nnz > 0) product = src1.entries[0].getZero();
        else if(src2.nnz > 0) product = src2.entries[0].getZero();

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.entries.length && src2Counter < src2.entries.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                product = product.add(src1.entries[src1Counter].mult((T) src2.entries[src2Counter]));
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return product;
    }


    /**
     * Computes the vector outer product between two complex sparse vectors.
     * @param src1 Entries of the first sparse vector in the outer product.
     * @param src2 Second sparse vector in the outer product.
     * @return The matrix resulting from the vector outer product.
     */
    public static <T extends Field<T>> DenseFieldMatrixBase<?, ?, ?, ?, T> outerProduct(
            CooFieldVectorBase<?, ?, ?, ?, T> src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        Field<T>[] dest = new Field[src2.size*src1.size];
        Arrays.fill(dest, src1.getZeroElement());

        int destRow;
        int index1;
        int index2;

        for(int i=0; i<src1.nnz; i++) {
            index1 = src1.indices[i];
            destRow = index1*src1.size;
            Field<T> src1Val = src1.entries[i];

            for(int j=0; j<src2.nnz; j++) {
                index2 = src2.indices[j];
                dest[destRow + index2] = src1Val.mult((T) src2.entries[j]);
            }
        }

        return src1.makeLikeDenseMatrix(new Shape(src1.size, src2.size), (T[]) dest);
    }



    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param src1 First vector in the stack.
     * @param src2 Vector to stack to the bottom of the {@code src2} vector.
     * @return The result of stacking this vector and vector {@code src2}.
     * @throws IllegalArgumentException If the number of entries in the {@code src1} vector is different from the number of entries in
     *                                  the vector {@code src2}.
     */
    public static <T extends Field<T>> CooFieldMatrixBase<?, ?, ?, ?, T> stack(CooFieldVectorBase<?, ?, ?, ?, T> src1,
                                                                               CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        Field<T>[] entries = new Field[src1.entries.length + src2.entries.length];
        int[][] indices = new int[2][src1.indices.length + src2.indices.length]; // Row and column indices.

        // Copy values from vector src1.
        System.arraycopy(src1.entries, 0, entries, 0, src1.entries.length);
        // Copy values from vector src2.
        System.arraycopy(src2.entries, 0, entries, src1.entries.length, src2.entries.length);

        // Set row indices to 1 for src2 values (this vectors row indices are 0 which was implicitly set already).
        Arrays.fill(indices[0], src1.indices.length, entries.length, 1);

        // Copy indices from src1 vector to the column indices.
        System.arraycopy(src1.indices, 0, indices[1], 0, src1.entries.length);
        // Copy indices from src2 vector to the column indices.
        System.arraycopy(src2.indices, 0, indices[1], src1.entries.length, src2.entries.length);

        return src1.makeLikeMatrix(new Shape(2, src1.size), (T[]) entries, indices[0], indices[1]);
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param src The vector to repeat.
     * @param n Number of times to repeat vector.
     * @param axis Axis along which to repeat vector. If {@code axis=0} then each row of the resulting matrix will be equivalent to
     * this vector. If {@code axis=1} then each column of the resulting matrix will be equivalent to this vector.
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    public static <T extends Field<T>> CooFieldMatrixBase<?, ?, ?, ?, T> repeat(CooFieldVectorBase<?, ?, ?, ?, T> src,
                                                                                int n,
                                                                                int axis) {
        ValidateParameters.ensureInRange(axis, 0, 1, "axis");
        ValidateParameters.ensureGreaterEq(0, n, "n");

        Shape tiledShape;
        Field<T>[] tiledEntries = new Field[n*src.entries.length];
        int[] tiledRows = new int[tiledEntries.length];
        int[] tiledCols = new int[tiledEntries.length];
        int nnz = src.nnz;

        if(axis==0) {
            tiledShape = new Shape(n, src.size);

            for(int i=0; i<n; i++) { // Copy values into row and set col indices as vector indices.
                System.arraycopy(src.entries, 0, tiledEntries, i*nnz, nnz);
                System.arraycopy(src.indices, 0, tiledCols, i*nnz, src.indices.length);
                Arrays.fill(tiledRows, i*nnz, (i+1)*nnz, i);
            }
        } else {
            int[] colIndices = ArrayUtils.intRange(0, n);
            tiledShape = new Shape(src.size, n);

            for(int i=0; i<nnz; i++) {
                Arrays.fill(tiledEntries, i*n, (i+1)*n, src.entries[i]);
                Arrays.fill(tiledRows, i*n, (i+1)*n, src.indices[i]);
                System.arraycopy(colIndices, 0, tiledCols, i*n, n);
            }
        }

        return src.makeLikeMatrix(tiledShape, (T[]) tiledEntries, tiledRows, tiledCols);
    }

}
