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

package org.flag4j.core_temp.arrays.dense;

import org.flag4j.core.Shape;
import org.flag4j.core_temp.arrays.sparse.CooFieldMatrix;
import org.flag4j.core_temp.arrays.sparse.CsrFieldMatrix;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.util.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A dense matrix whose entries are {@link Field} elements.</p>
 *
 * <p>Field matrices have mutable entries but fixed shape.</p>
 *
 * <p>A matrix is essentially equivalent to a rank 2 tensor but has some extended functionality and may have improved performance
 * for some operations.</p>
 *
 * @param <T> Type of the {@link Field field} element for the matrix.
 */
public class FieldMatrix<T extends Field<T>> extends DenseFieldMatrixBase<FieldMatrix<T>, CooFieldMatrix<T>,
        CsrFieldMatrix<T>, FieldVector<T>, T> {


    /**
     * Creates a dense field matrix with the specified entries and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(Shape shape, T[] entries) {
        super(shape, entries);
    }


    /**
     * Creates a dense field matrix with the specified entries and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(int rows, int cos, T[] entries) {
        super(new Shape(rows, cos), entries);
    }



    /**
     * Creates a dense field matrix with the specified entries and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(Shape shape, T[][] entries) {
        super(shape, ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense field matrix with the specified entries and filled with {@code filledValue}.
     *
     * @param shape Shape of this matrix.
     * @param fillValue Entries of this matrix.
     */
    public FieldMatrix(Shape shape, T fillValue) {
        super(shape, (T[]) new Field[shape.totalEntriesIntValueExact()]);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a dense field matrix with the specified entries and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(int rows, int cols, T[][] entries) {
        super(new Shape(rows, cols), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense field matrix with the specified entries and filled with {@code filledValue}.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param fillValue Entries of this matrix.
     */
    public FieldMatrix(int rows, int cols, T fillValue) {
        super(new Shape(rows, cols), (T[]) new Field[rows*cols]);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public FieldMatrix<T> makeLikeTensor(Shape shape, T[] entries) {
        return new FieldMatrix<T>(shape, entries);
    }


    /**
     * Constructs a matrix of the same type as this matrix with the given the shape filled with the specified fill value.
     *
     * @param shape Shape of the matrix to construct.
     * @param fillValue Value to fill this matrix with.
     *
     * @return A matrix of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public FieldMatrix<T> makeLikeTensor(Shape shape, T fillValue) {
        return new FieldMatrix<T>(shape, fillValue);
    }


    /**
     * Constructs a vector of similar type to this matrix with the given {@code entries}.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of similar type to this matrix with the given {@code entries}.
     */
    @Override
    public FieldVector<T> makeLikeVector(T... entries) {
        return new FieldVector<T>(entries);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooFieldMatrix<T> toCoo() {
        int rows = numRows;
        int cols = numCols;
        List<T> sparseEntries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                T val = entries[rowOffset + j];

                if(!val.isZero()) {
                    sparseEntries.add(val);
                    rowIndices.add(i);
                    colIndices.add(j);
                }
            }
        }

        return new CooFieldMatrix<T>(shape, sparseEntries, rowIndices, colIndices);
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link Matrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldMatrix<T> src2 = (FieldMatrix<T>) object;

        return shape.equals(src2.shape) && Arrays.equals(entries, src2.entries);
    }
}
