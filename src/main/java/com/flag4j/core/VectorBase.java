package com.flag4j.core;

import com.flag4j.Shape;


/**
 * The base class for all vectors.
 * @param <T> The type of entries for this matrix.
 */
public abstract class VectorBase<T> extends TensorBase<T> {

    VectorOrientations orientation;

    /**
     * Constructs a basic vector with the specified number of entries.
     * @param size Number of entries in this vector.
     * @param orientation Orientation of this vector.
     */
    public VectorBase(int size, VectorOrientations orientation, T entries) {
        super(new Shape(size), entries);
        this.orientation = orientation;
    }


    /**
     * Gets the size of this vector.
     * @return The size, i.e. number of entries, of this vector.
     */
    public int size() {
        return super.totalEntries().intValue();
    }


    /**
     * Gets the oriented shape of this vector.
     * @return The oriented shape of this vector. <br>
     * If this vector is a {@link VectorOrientations#ROW row} vector, then
     * the shape will be {@code (1, this.size())}. <br>
     * If this vector is a {@link VectorOrientations#COL column} vector or an {@link VectorOrientations#UNORIENTED unoriented}
     * vector then the shape will be {@code (1, this.size())}.
     */
    public Shape getOrientedShape() {
        Shape orientedShape;

        if(this.orientation==VectorOrientations.ROW) {
            orientedShape = new Shape(1, this.size());
        } else {
            orientedShape = new Shape(this.size(), 1);
        }

        return orientedShape;
    }


    /**
     * Gets the orientation of this vector.
     * @return The orientation of this vector.
     */
    public VectorOrientations getOrientation() {
        return this.orientation;
    }
}
