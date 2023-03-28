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

package com.flag4j.core;

import com.flag4j.Shape;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;

import java.io.Serializable;


/**
 * <p>
 *     The base class for all matrices. A matrix is equivalent to a {@link TensorBase tensor} of rank 2.
 * </p>
 *
 * @param <T> Type of the storage data structure for the matrix.
 *           This common use case will be an array or list-like data structure.
 */
public abstract class MatrixBase<T extends Serializable> extends TensorBase<T> {

    // TODO: Move DEFAULT_ROUND_TO_ZERO_THRESHOLD somewhere else and use for all tensors.
    /**
     * Default value for rounding to zero.
     */
    protected static final double DEFAULT_ROUND_TO_ZERO_THRESHOLD = 1e-12;
    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;


    /**
     * Constructs a basic matrix with a given shape.
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     * @throws IllegalArgumentException If the shape parameter is not of rank 2.
     */
    public MatrixBase(Shape shape, T entries) {
        super(shape, entries);

        numRows = shape.dims[Axis2D.row()];
        numCols = shape.dims[Axis2D.col()];

        if(shape.getRank() != 2) {
            throw new IllegalArgumentException(ErrorMessages.shapeRankErr(shape.getRank(), 2));
        }
    }


    /**
     * Gets the number of rows in this matrix.
     * @return The number of rows in this matrix.
     */
    public int numRows() {
        return numRows;
    }


    /**
     * Gets the number of columns in this matrix.
     * @return The number of columns in this matrix.
     */
    public int numCols() {
        return numCols;
    }


    public abstract MatrixBase<T> flatten(int axis);
}