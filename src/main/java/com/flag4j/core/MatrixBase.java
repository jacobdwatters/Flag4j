package com.flag4j.core;

import com.flag4j.Shape;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;


/**
 * The base class for all matrices.
 * @param <T> The type of entries for this matrix.
 */
public abstract class MatrixBase<T> extends TensorBase<T> {

    /**
     * Constructs a basic matrix with a given shape.
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     * @throws IllegalArgumentException If the shape parameter is not of rank 2.
     */
    public MatrixBase(Shape shape, T entries) {
        super(shape, entries);

        if(shape.getRank() != 2) {
            throw new IllegalArgumentException(ErrorMessages.shapeRankErr(shape.getRank(), 2));
        }
    }


    /**
     * Gets the number of rows in this matrix.
     * @return The number of rows in this matrix.
     */
    public int numRows() {
        return shape.dims[Axis2D.row()];
    }


    /**
     * Gets the number of columns in this matrix.
     * @return
     */
    public int numCols() {
        return shape.dims[Axis2D.col()];
    }
}